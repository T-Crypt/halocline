// tests/kernels/gqa_decode_bench.cpp - decode attention timing on one layer's K8V4 cache at the served dense geometry
// (24 query heads, 4 KV heads, 256 wide): the engine's kernel (decode_k8v4_kernel, a window of M = 6T <= 48 rows in one
// block, its chunks' raw bytes staged with cp.async) beside the two the engine does not select (the window kernel,
// gqa_decode_attn_window, and the 16-row-tile launch, test::gqa_decode_attn_tiled), T = 1..8 rows ending at cell DEPTH.
// A timing tool, not a ctest: run it by hand under a GPU lease.  The cache bytes are zero (finite, and the time does not
// depend on the values).
//
//     gqa_decode_bench [DEPTH=137404] [REPS=50]
#include "halo/kernels/gqa_attn.hpp"
#include "halo/kernels/kv_q4_layout.hpp"

#include "gqa_decode_ref.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace k = halo::kernels;

int main(int argc, char** argv) {
    const int64_t depth = argc > 1 ? std::atoll(argv[1]) : 137404, max_cells = 175616;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    constexpr int NH = 24, NKV = 4, HD = 256;
    int8_t* d_k; uint16_t* d_ks; uint8_t* d_vq; float *d_q, *d_out, *d_scr;
    const size_t cells = (size_t) max_cells * NKV;
    if (cudaMalloc(&d_k, cells * HD) != cudaSuccess) { std::puts("no GPU"); return 77; }
    cudaMalloc(&d_ks, cells * (HD / 64) * 2);
    cudaMalloc(&d_vq, cells * k::kv_q4_head_bytes(HD));
    cudaMemset(d_k, 0, cells * HD); cudaMemset(d_ks, 0, cells * (HD / 64) * 2); cudaMemset(d_vq, 0, cells * k::kv_q4_head_bytes(HD));
    cudaMalloc(&d_q, (size_t) 16 * NH * HD * 4); cudaMemset(d_q, 0, (size_t) 16 * NH * HD * 4);
    cudaMalloc(&d_out, (size_t) 16 * NH * HD * 4);
    cudaMalloc(&d_scr, k::gqa_decode_scratch_floats(max_cells, 16, NH) * 4);
    k::KvK8V4 cache{d_k, d_ks, d_vq};
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::printf("depth %lld, %d reps, ms per launch (k8v4 / window kernel / 16-row tiles), GB/s of one KV pass\n",
                (long long) depth, reps);
    for (int T = 1; T <= 8; ++T) {
        const int64_t p0 = depth - T + 1;
        float ms[3];
        for (int which = 0; which < 3; ++which) {
            auto run = [&] {
                if (which == 0) k::gqa_decode_attn(d_q, nullptr, nullptr, NH, NKV, p0, T, d_scr, max_cells, d_out, nullptr, nullptr,
                                                   k::KvInt8{}, nullptr, nullptr, k::KvFar{}, cache);
                else if (which == 1) k::test::gqa_decode_attn_window(d_q, nullptr, nullptr, NH, NKV, p0, T, d_scr, max_cells, d_out,
                                                                     nullptr, nullptr, k::KvInt8{}, nullptr, nullptr, k::KvFar{},
                                                                     cache);
                else k::test::gqa_decode_attn_tiled(d_q, nullptr, nullptr, NH, NKV, p0, T, d_scr, max_cells, d_out, nullptr,
                                                    nullptr, k::KvInt8{}, nullptr, nullptr, k::KvFar{}, cache);
            };
            for (int i = 0; i < 3; ++i) run();
            cudaEventRecord(a);
            for (int i = 0; i < reps; ++i) run();
            cudaEventRecord(b); cudaEventSynchronize(b);
            cudaEventElapsedTime(&ms[which], a, b);
            ms[which] /= reps;
        }
        const double gb = (double) depth * NKV * (HD + HD / 64 * 2 + k::kv_q4_head_bytes(HD)) / 1e9;
        std::printf("T %d  k8v4 %.3f ms (%.0f GB/s)  window %.3f ms (%.0f)  tiles %.3f ms (%.0f one pass)\n", T, ms[0],
                    gb / ms[0] * 1e3, ms[1], gb / ms[1] * 1e3, ms[2], gb / ms[2] * 1e3);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
