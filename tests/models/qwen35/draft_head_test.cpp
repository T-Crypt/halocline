// tests/models/qwen35/draft_head_test.cpp - --draft-vocab's gather, on a card: the draft head's rows are the head's
// rows byte for byte, the subset head's one-row product gives the full head's logits at those ids bit for bit (same
// kernel, same rows, same activations), and the row map takes a subset index back to its token id.  A synthetic Q6_K
// head, no pack: the head's shape is the only thing this claims anything about.  Exit 77 without a GPU.
#include "halo/dense/model.hpp"
#include "halo/kernels/head_rows.hpp"
#include "halo/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// a Q6_K row's bytes with every block's fp16 scale set to 1.0, so the product is finite and comparable: 210 bytes a
// block, the scale the last two (ql[128], qh[64], scales[16], d)
std::vector<uint8_t> q6k_rows(int64_t rows, int64_t cols) {
    const int64_t blocks = cols / 256;
    std::vector<uint8_t> v((size_t) (rows * blocks * 210), 0x11);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < blocks; ++b) {
            uint8_t* blk = v.data() + (size_t) ((r * blocks + b) * 210);
            blk[208] = 0x00;   // fp16 1.0 little-endian
            blk[209] = 0x3c;
        }
    return v;
}
}  // namespace

int main() {
    using namespace halo::dense;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    constexpr int64_t kCols = 5120, kRows = 256;
    constexpr size_t kRowBytes = 4200;   // Q6_K, 5120 values
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) return 77;

    // the source head, as the pack's load leaves it on the card
    const std::vector<uint8_t> src = q6k_rows(kRows, kCols);
    uint8_t* d_src = nullptr;
    if (cudaMalloc((void**) &d_src, src.size()) != cudaSuccess ||
        cudaMemcpyAsync(d_src, src.data(), src.size(), cudaMemcpyHostToDevice, s) != cudaSuccess) {
        std::printf("FAIL no device memory for the test head\n");
        return 1;
    }
    Linear full;
    full.type = 14;   // Q6_K
    full.w = d_src;
    full.rows = kRows;
    full.cols = kCols;

    // the gather: ids out of order (the map is what carries the order), and the map back
    const std::vector<int32_t> ids = {7, 3, 200, 255, 0, 128};
    Linear sub;
    const int32_t* map = nullptr;
    std::string err;
    const bool gathered = gather_draft_head(full, ids, s, sub, map, err);
    check(gathered, "the draft head gathers: " + err);
    if (!gathered) return 1;
    check(draft_head_bytes(full, (int64_t) ids.size()) == (uint64_t) kRowBytes * ids.size(),
          "the draft head's bytes are the rows it names");
    check(sub.type == full.type && sub.rows == (int64_t) ids.size() && sub.cols == kCols,
          "the draft head keeps the head's type and width, with the subset's rows");

    std::vector<uint8_t> got((size_t) kRowBytes * ids.size());
    cudaMemcpyAsync(got.data(), sub.w, got.size(), cudaMemcpyDeviceToHost, s);
    std::vector<int32_t> got_map(ids.size());
    cudaMemcpyAsync(got_map.data(), map, ids.size() * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    bool rows_ok = true;
    for (size_t i = 0; i < ids.size(); ++i)
        rows_ok = rows_ok && std::memcmp(got.data() + i * kRowBytes, src.data() + (size_t) ids[i] * kRowBytes, kRowBytes) == 0;
    check(rows_ok, "every gathered row is the source row's bytes, exactly");
    check(got_map == ids, "the id map holds the file's ids, row by row");

    // one row's product through the same kernel: the subset's logit i is the full head's logit ids[i], bit for bit
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) kCols);
    for (float& v : x) v = nd(rng);
    float* d_x = nullptr;
    void* xq = nullptr;
    float* y_full = nullptr;
    float* y_sub = nullptr;
    if (cudaMalloc((void**) &d_x, x.size() * 4) != cudaSuccess || cudaMalloc(&xq, (size_t) kCols / 32 * 36) != cudaSuccess ||
        cudaMalloc((void**) &y_full, kRows * 4) != cudaSuccess || cudaMalloc((void**) &y_sub, ids.size() * 4) != cudaSuccess) {
        std::printf("FAIL no device memory for the test's activations\n");
        return 1;
    }
    cudaMemcpyAsync(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s);
    halo::kernels::native_quantize_q8_1(d_x, xq, (int) kCols, 1, s);
    halo::kernels::native_mmvq(full.type, full.w, xq, y_full, (int) kCols, (int) kRows, 1, s);
    halo::kernels::native_mmvq(sub.type, sub.w, xq, y_sub, (int) kCols, (int) sub.rows, 1, s);
    std::vector<float> h_full((size_t) kRows), h_sub(ids.size());
    cudaMemcpyAsync(h_full.data(), y_full, (size_t) kRows * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_sub.data(), y_sub, ids.size() * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    bool logits_ok = true;
    for (size_t i = 0; i < ids.size(); ++i)
        logits_ok = logits_ok && std::memcmp(&h_sub[i], &h_full[(size_t) ids[i]], sizeof(float)) == 0;
    check(logits_ok, "the subset head's logits are the full head's at those ids, bit for bit");

    // the drafter's remap: a subset row index becomes the token id the next draft step embeds
    int32_t* d_pick = nullptr;
    cudaMalloc((void**) &d_pick, 4);
    const int32_t local = 4;   // ids[4] = 0
    cudaMemcpyAsync(d_pick, &local, 4, cudaMemcpyHostToDevice, s);
    halo::kernels::map_ids(d_pick, map, 1, s);
    int32_t back = -1;
    cudaMemcpyAsync(&back, d_pick, 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    check(back == ids[(size_t) local], "map_ids turns a draft head row into its token id");

    // a head the drafter's product cannot read, and one with no rows, are refused by name
    {
        Linear odd = full;
        odd.type = 1;   // F16: not a GGUF quant the one-row product reads
        Linear unused;
        const int32_t* unused_map = nullptr;
        std::string why;
        check(!gather_draft_head(odd, ids, s, unused, unused_map, why), "a head of another type is refused");
        check(why.find("does not read natively") != std::string::npos, "and the error names the product's limit");
        Linear none;
        check(!gather_draft_head(none, ids, s, unused, unused_map, why), "a head with no rows is refused");
    }

    cudaFree(d_pick);
    cudaFree(y_sub);
    cudaFree(y_full);
    cudaFree(xq);
    cudaFree(d_x);
    cudaFree((void*) sub.w);
    cudaFree((void*) map);
    cudaFree(d_src);
    cudaStreamDestroy(s);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
