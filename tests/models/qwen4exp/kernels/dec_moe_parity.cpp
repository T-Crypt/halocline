// tests/models/qwen4exp/kernels/dec_moe_parity.cpp - DEC_MOE's kernels on real GGUF experts against the path they replace.
//
//     build/dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench]
//     build/dec_moe_parity --nvfp4 [--bench]     (synthetic NVFP4 experts with their scale tail, no GGUF)
//     build/dec_moe_parity --kq [--bench]        (synthetic Q4_K / Q5_K / Q6_K gate/up with Q5_1 / Q8_0 down, and
//                                                 IQ2_XXS / IQ2_S / IQ3_XXS / IQ4_XS gate/up with Q5_0 down, no GGUF)
//
// Per layer (default: one of each gate/up + down format pair):
//   route       dec_moe_route's plan, kinds, q8_1 rows (vs quantize_q8_1_rows) and mapped publications
//   tail        route + native_expert_grouped + dec_moe_combine vs the old chain (grouped into hit rows, the mapped
//               CPU rows copied, hit add, native_moe_combine_multi)
//   staged      the share of CPU misses copied from host RAM into staging: the route's, the ring's, the
//               glue's and the stage kernel's kinds vs a CPU reference of the split, the staged lists and bytes, and the
//               tail against the same window with those experts resident in VRAM
// All bitwise.  --bench adds: the VRAM share's grouped call (us, GB/s of expert bytes), the combine's launches, and the
// staged share's copy (us, GB/s over the link).
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/native_moe.hpp"
#include "basalt/models/qwen4exp/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace K = basalt::kernels;

namespace {

int g_fail = 0;
constexpr int64_t H = 2560, FF = 640, NE = 512, KK = 10, MT = 8;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template<typename T> T* dmal(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    ck(cudaDeviceSynchronize(), "alloc sync");   // the tests' streams are non-blocking
    return p;
}
template<typename T> T* hmap(size_t n) {
    T* p = nullptr;
    ck(cudaHostAlloc((void**) &p, n * sizeof(T) + 256, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    std::memset((void*) p, 0, n * sizeof(T) + 256);
    return p;
}
template<typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
void report(const char* what, int layer, size_t ndiff, size_t n) {
    std::printf("  %-34s layer %2d: %s (%zu of %zu differ)\n", what, layer, ndiff ? "DIFFERS" : "bitwise equal", ndiff, n);
    if (ndiff) ++g_fail;
}
size_t ndiff_rows(const std::vector<float>& a, const std::vector<float>& b, const std::vector<int>& rows) {
    size_t d = 0;
    for (int r : rows)
        for (int64_t c = 0; c < H; ++c) d += std::memcmp(&a[(size_t) r * H + c], &b[(size_t) r * H + c], 4) != 0;
    return d;
}

// A UE4M3 sub-block scale byte.  Profile 0: exponent 1..4 (small blocks).  Profile 1: exponent 6..14, block values in
// the hundreds as the packs' are (their tails are 1e-5..1e-4).  Profile 2: profile 1 with the odd bytes mixed in: 0x7F
// (the NaN code, decoded as 0), the sign bit set (ignored), zero and subnormals.
uint8_t nvfp4_scale_byte(int prof, std::mt19937& rng) {
    if (prof == 0) return (uint8_t) (((1 + rng() % 4) << 3) | (rng() & 7));
    const uint8_t v = (uint8_t) (((6 + rng() % 9) << 3) | (rng() & 7));
    if (prof == 1) return v;
    switch (rng() % 8) {
        case 0: return 0x7F;
        case 1: return (uint8_t) (v | 0x80);
        case 2: return 0xFF;
        case 3: return 0x00;
        case 4: return (uint8_t) (rng() & 7);   // subnormal
        default: return v;
    }
}
// The FP32 tail scale: profile 0 in 0.5 .. 2, else log-uniform in 2e-5 .. 2e-4 (what the packs carry).
float nvfp4_tail_scale(int prof, std::mt19937& rng) {
    if (prof == 0) return std::uniform_real_distribution<float>(0.5f, 2.0f)(rng);
    return 2e-5f * std::exp(std::uniform_real_distribution<float>(0.0f, std::log(10.0f))(rng));
}
// NX synthetic NVFP4 expert blobs: random nibbles, UE4M3 sub-block scales, then the FP32 tail {s_gate, s_up, s_down, 0};
// expert x takes scale profile x % 3
std::vector<uint8_t> nvfp4_blobs(const K::NativeExpertLayout& L, int nx, std::mt19937& rng) {
    std::vector<uint8_t> b((size_t) nx * L.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    constexpr size_t bs = 36;   // block_nvfp4: 4 scale bytes, 32 nibble bytes
    for (int x = 0; x < nx; ++x) {
        uint8_t* p = b.data() + (size_t) x * L.bytes;
        const size_t q = L.bytes - 16;
        for (size_t i = 0; i < q; ++i) p[i] = (uint8_t) byte(rng);
        for (size_t o = 0; o < q; o += bs)
            for (int j = 0; j < 4; ++j) p[o + j] = nvfp4_scale_byte(x % 3, rng);
        const float t[4] = {nvfp4_tail_scale(x % 3, rng), nvfp4_tail_scale(x % 3, rng), nvfp4_tail_scale(x % 3, rng), 0.0f};
        std::memcpy(p + q, t, sizeof t);
    }
    return b;
}

// NX synthetic blobs of the K-quant, Q5_0 / Q8_0 and i-quant formats (Q4_K / Q5_K / Q8_0 / IQ gate/up, Q5_1 / Q5_0 / Q8_0
// down): random bytes, then finite small fp16 block scales (the first two halves of a Q4_K / Q5_K / Q5_1 block, the
// first half of the others; IQ3_S / IQ4_XS keep d in the block's first two bytes)
void kq_fix_scales(uint8_t* p, size_t bytes, int type, std::mt19937& rng) {
    const size_t bs = type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 210 : type == 7 ? 24 : type == 6 ? 22
                    : type == 16 ? 66 : type == 22 ? 82 : type == 18 ? 98 : type == 21 ? 110 : type == 23 ? 136 : 34;
    std::uniform_int_distribution<int> ex(2, 5), man(0, 1023);
    for (size_t o = 0; o + bs <= bytes; o += bs) {
        if (type == 14) {   // Q6_K: the fp16 scale ends the block
            const uint16_t h = (uint16_t) ((ex(rng) << 10) | man(rng));
            std::memcpy(p + o + 208, &h, 2);
            continue;
        }
        for (int j = 0; j < (type == 12 || type == 13 || type == 7 ? 2 : 1); ++j) {
            const uint16_t h = (uint16_t) ((ex(rng) << 10) | man(rng));
            std::memcpy(p + o + 2 * j, &h, 2);
        }
    }
}
std::vector<uint8_t> kq_blobs(const K::NativeExpertLayout& L, int nx, std::mt19937& rng) {
    std::vector<uint8_t> b((size_t) nx * L.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& v : b) v = (uint8_t) byte(rng);
    for (int x = 0; x < nx; ++x) {
        uint8_t* p = b.data() + (size_t) x * L.bytes;
        kq_fix_scales(p, L.down_off, L.gu_type, rng);
        kq_fix_scales(p + L.down_off, L.bytes - L.down_off, L.d_type, rng);
    }
    return b;
}

struct Groups {   // a grouped call's lists on one device
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_n = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    void upload() {
        const int ng = (int) ptr.size();
        d_ptr = dmal<unsigned long long>(K::kDmCap);
        d_start = dmal<int32_t>(K::kDmCap + 1);
        d_n = dmal<int32_t>(4);
        d_dst = dmal<int32_t>(K::kDmCap);
        d_tok = dmal<int32_t>(K::kDmCap);
        ck(cudaMemcpy(d_ptr, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_start, start.data(), start.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_n, &ng, 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_dst, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(d_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice), "up");
    }
};

double time_us(cudaStream_t s, int iters, const std::function<void()>& f) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 5; ++i) f();
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < iters; ++i) f();
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        best = std::min(best, 1e3 * ms / iters);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: dec_moe_parity <shard1.gguf> [layer,layer,...] [--bench]\n       dec_moe_parity --nvfp4 [--bench]\n"
                     "       dec_moe_parity --kq [--bench]\n");
        return 2;
    }
    bool bench = false;
    std::vector<int> layers;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) { bench = true; continue; }
        std::stringstream ss(argv[i]);
        std::string t;
        while (std::getline(ss, t, ',')) layers.push_back(std::atoi(t.c_str()));
    }
    const bool nvfp4 = std::strcmp(argv[1], "--nvfp4") == 0, kq = std::strcmp(argv[1], "--kq") == 0;
    const bool synth = nvfp4 || kq;
    std::unique_ptr<basalt::GgufFile> gguf_p;
    if (!synth) gguf_p = std::make_unique<basalt::GgufFile>(argv[1]);
    int ndev = 0;
    ck(cudaGetDeviceCount(&ndev), "device count");
    const bool two = ndev >= 2;
    std::printf("dec_moe_parity: %d device(s)%s\n", ndev, two ? " (window test on device 1)" : "");
    // the layers' formats; by default the first layer of each pair
    std::vector<std::pair<int, int>> fmt(64, {-1, -1});
    std::vector<int> auto_layers;
    if (nvfp4) { fmt[0] = {40, 40}; auto_layers.push_back(0); layers.clear(); }
    if (kq) {
        fmt[0] = {12, 7}; fmt[1] = {12, 8}; fmt[2] = {13, 8}; fmt[3] = {8, 8}; fmt[4] = {14, 8};
        fmt[5] = {16, 6}; fmt[6] = {22, 6}; fmt[7] = {18, 6}; fmt[8] = {23, 6};
        fmt[9] = {21, 8}; fmt[10] = {23, 8};
        auto_layers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    }
    for (int l = 0; !synth && l < 64; ++l) {
        const basalt::GgufFile& gguf = *gguf_p;
        const auto* g = gguf.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
        const auto* d = gguf.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
        if (!g || !d) continue;
        fmt[l] = {(int) g->type, (int) d->type};
        bool seen = false;
        for (int a : auto_layers) seen |= fmt[a] == fmt[l];
        if (!seen) auto_layers.push_back(l);
    }
    if (layers.empty()) layers = auto_layers;
    cudaStream_t s0, s1 = nullptr;
    ck(cudaSetDevice(0), "dev0");
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    if (two) {
        ck(cudaSetDevice(1), "dev1");
        ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
        ck(cudaSetDevice(0), "dev0");
    }
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);

    for (int l : layers) {
        if (l < 0 || l >= 64 || fmt[l].first < 0) { std::printf("layer %d: no experts\n", l); ++g_fail; continue; }
        const int gt = fmt[l].first, dt = fmt[l].second;
        const auto L = K::native_expert_layout(gt, dt, H, FF);
        std::printf("layer %d: gate/up type %d, down type %d, %.2f MB per expert, formats %s\n", l, gt, dt,
                    L.bytes / 1e6, K::dec_moe_formats_ok(gt, dt, H, FF) ? "covered" : "NOT COVERED");
        if (!K::dec_moe_formats_ok(gt, dt, H, FF)) { ++g_fail; continue; }
        // NX distinct experts' blobs; expert id e uses blob e % NX
        constexpr int NX = 48;
        std::vector<uint8_t> blobs = nvfp4 ? nvfp4_blobs(L, NX, rng) : kq ? kq_blobs(L, NX, rng)
                                                                    : std::vector<uint8_t>((size_t) NX * L.bytes);
        for (int x = 0; !synth && x < NX; ++x) {
            const basalt::GgufFile& gguf = *gguf_p;
            const auto* tg = gguf.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight");
            const auto* tu = gguf.find("blk." + std::to_string(l) + ".ffn_up_exps.weight");
            const auto* td = gguf.find("blk." + std::to_string(l) + ".ffn_down_exps.weight");
            const size_t E = (size_t) ((x * 37 + 5) % NE);
            uint8_t* b = blobs.data() + (size_t) x * L.bytes;
            std::memcpy(b, gguf.tensor_data(*tg) + E * L.up_off, L.up_off);
            std::memcpy(b + L.up_off, gguf.tensor_data(*tu) + E * L.up_off, L.up_off);
            std::memcpy(b + L.down_off, gguf.tensor_data(*td) + E * (L.bytes - L.down_off), L.bytes - L.down_off);
        }
        std::vector<float> x((size_t) MT * H);
        for (auto& v : x) v = nd(rng);

        struct Dev {
            uint8_t* blob = nullptr;
            float* x = nullptr;
            void* xq = nullptr;
            void* scr = nullptr;
            float* out = nullptr;
        } dv[2];
        for (int d = 0; d < (two ? 2 : 1); ++d) {
            ck(cudaSetDevice(d), "dev");
            dv[d].blob = dmal<uint8_t>(blobs.size());
            ck(cudaMemcpy(dv[d].blob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "blobs");
            dv[d].x = dmal<float>(x.size());
            ck(cudaMemcpy(dv[d].x, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            dv[d].xq = dmal<uint8_t>((size_t) MT * H / 32 * 36);
            K::quantize_q8_1_rows(dv[d].x, MT, H, dv[d].xq, nullptr);
            dv[d].scr = dmal<uint8_t>(K::native_expert_scratch_bytes(K::kDmCap, FF));
            dv[d].out = dmal<float>((size_t) K::kDmCap * H);
        }
        ck(cudaSetDevice(0), "dev0");

        // ---- route + tail on CUDA0, T = 1, 3, 8
        int32_t* d_res0 = dmal<int32_t>(NE);
        std::vector<int32_t> res0(NE);
        for (int e = 0; e < NE; ++e) {
            const int r = (int) (rng() % 8);
            res0[e] = r < 6 ? e % NX : -1;            // the slot = the blob
        }
        ck(cudaMemcpy(d_res0, res0.data(), NE * 4, cudaMemcpyHostToDevice), "res");
        K::DmPlan* d_plan = dmal<K::DmPlan>(1);
        void* d_xq2 = dmal<uint8_t>((size_t) MT * H / 32 * 36);
        int32_t* d_ids = dmal<int32_t>(K::kDmCap);
        float* d_w = dmal<float>(K::kDmCap);
        float* d_sh = dmal<float>((size_t) MT * H);
        float* d_bo = dmal<float>((size_t) MT * H);
        float* d_bo2 = dmal<float>((size_t) MT * H);
        float* d_parts = dmal<float>((size_t) K::kDmCap * H);
        float* d_hit = dmal<float>((size_t) K::kDmCap * H);
        unsigned int* d_ctr = dmal<unsigned int>(4);
        void* d_vscr = dmal<uint8_t>(K::dec_moe_vram_scratch_bytes((int) FF));
        unsigned int* d_vsync = dmal<unsigned int>(K::kDmCap + 1);
        constexpr int RINGS = 48;
        K::DmRing* h_rings = hmap<K::DmRing>(RINGS);
        float* h_x = hmap<float>((size_t) RINGS * MT * H);
        uint8_t* h_xq = hmap<uint8_t>((size_t) RINGS * MT * H / 32 * 36);
        uint32_t* h_words = hmap<uint32_t>(64);   // [0] host seq [48] host flag
        float* h_cpu = hmap<float>((size_t) K::kDmCap * H);
        float* h_ymiss = hmap<float>((size_t) K::kDmCap * H);
        const size_t xq_stride = (size_t) MT * H / 32 * 36;
        for (int T : {1, 3, 8}) {
            const int n = T * (int) KK;
            std::vector<int32_t> ids(n);
            for (int t = 0; t < T; ++t) {
                std::vector<int> pick;
                while ((int) pick.size() < KK) {
                    const int e = (int) (rng() % 64);   // a small pool: tokens share experts
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int j = 0; j < KK; ++j) ids[t * KK + j] = pick[j];
            }
            std::vector<float> w(n), sh((size_t) T * H), cpu((size_t) n * H);
            for (auto& v : w) v = std::fabs(nd(rng)) * 0.1f;
            for (auto& v : sh) v = nd(rng);
            for (auto& v : cpu) v = nd(rng);
            ck(cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice), "ids");
            ck(cudaMemcpy(d_w, w.data(), n * 4, cudaMemcpyHostToDevice), "w");
            ck(cudaMemcpy(d_sh, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice), "sh");
            std::memcpy(h_cpu, cpu.data(), cpu.size() * 4);
            std::memset(h_words, 0, 256);
            K::DmRouteArgs ra;
            ra.ids = d_ids; ra.x = dv[0].x; ra.n_tok = T; ra.k = (int) KK; ra.n_embd = (int) H; ra.n_expert = (int) NE;
            ra.layer = l; ra.res0 = d_res0; ra.cache_base = dv[0].blob; ra.slot_off = nullptr;
            ra.blob = (long long) L.bytes; ra.plan = d_plan; ra.xq = d_xq2; ra.ring_m = h_rings + 5;
            ra.x_m = h_x + (size_t) 5 * MT * H; ra.xq_m = h_xq + 5 * xq_stride; ra.host_seq_m = h_words;
            ra.ring = 6; ra.ctr = d_ctr;
            K::dec_moe_route(ra, s0);
            ck(cudaStreamSynchronize(s0), "route");
            // reference: kinds, groups, xq
            std::vector<int32_t> kind(n);
            int nc = 0;
            for (int i = 0; i < n; ++i) {
                kind[i] = res0[ids[i]] >= 0 ? 0 : -1;
                nc += kind[i] < 0;
            }
            K::DmPlan P;
            ck(cudaMemcpy(&P, d_plan, sizeof P, cudaMemcpyDeviceToHost), "plan");
            size_t bad = 0;
            std::vector<int> seen(n, 0);
            for (int g = 0; g < P.counts[0]; ++g)
                for (int e = P.start[g]; e < P.start[g + 1]; ++e) {
                    const int i = P.dst[e];
                    bad += i < 0 || i >= n || kind[i] != 0 || P.tok[e] != i / KK ||
                           P.ptr[g] != (unsigned long long) (dv[0].blob + (size_t) res0[ids[i]] * L.bytes);
                    if (i >= 0 && i < n) ++seen[i];
                }
            for (int i = 0; i < n; ++i) bad += (kind[i] == 0) != (seen[i] == 1) || P.kind[i] != kind[i] ||
                                               h_rings[5].kind[i] != kind[i] || h_rings[5].ids[i] != ids[i];
            bad += P.counts[2] != nc || h_rings[5].n_cpu != nc || h_rings[5].n_tok != T || h_words[0] != 6;
            const auto q1 = down((const uint8_t*) dv[0].xq, (size_t) T * H / 32 * 36);
            const auto q2 = down((const uint8_t*) d_xq2, (size_t) T * H / 32 * 36);
            bad += std::memcmp(q1.data(), q2.data(), q1.size()) != 0;
            if (nc > 0) bad += std::memcmp(x.data(), h_x + (size_t) 5 * MT * H, (size_t) T * H * 4) != 0;
            char what[64];
            std::snprintf(what, sizeof what, "route T=%d (%d cpu)", T, nc);
            report(what, l, bad, (size_t) n);
            {   // the split route (mode 1 then mode 2) leaves every byte the one launch left
                const std::vector<uint8_t> ring0((const uint8_t*) (h_rings + 5), (const uint8_t*) (h_rings + 6));
                const std::vector<uint8_t> xm0((const uint8_t*) (h_x + (size_t) 5 * MT * H),
                                               (const uint8_t*) (h_x + (size_t) 5 * MT * H + (size_t) T * H));
                const uint32_t w0 = h_words[0];
                h_rings[5] = K::DmRing{};
                std::memset(h_x + (size_t) 5 * MT * H, 0, (size_t) T * H * 4);
                std::memset(h_xq + 5 * xq_stride, 0, q1.size());
                h_words[0] = 0;
                ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                ck(cudaMemsetAsync(d_xq2, 0xee, q1.size(), s0), "memset");
                void* d_pub = dmal<uint8_t>(xq_stride);
                K::DmRouteArgs r1 = ra, r2 = ra;
                r1.mode = 1; r2.mode = 2; r2.xq_pub = d_pub;
                K::dec_moe_route(r1, s0);
                K::dec_moe_route(r2, s0);
                ck(cudaStreamSynchronize(s0), "split route");
                K::DmPlan P2;
                ck(cudaMemcpy(&P2, d_plan, sizeof P2, cudaMemcpyDeviceToHost), "plan");
                const auto q3 = down((const uint8_t*) d_xq2, q1.size());
                size_t bad2 = std::memcmp(P.counts, P2.counts, sizeof P.counts) != 0;
                for (int g = 0; g <= P.counts[0]; ++g) bad2 += P.start[g] != P2.start[g] || (g < P.counts[0] && P.ptr[g] != P2.ptr[g]);
                for (int e = 0; e < P.counts[1]; ++e) bad2 += P.dst[e] != P2.dst[e] || P.tok[e] != P2.tok[e];
                for (int i = 0; i < n; ++i) bad2 += P.kind[i] != P2.kind[i];
                bad2 += std::memcmp(q3.data(), q2.data(), q2.size()) != 0;
                bad2 += std::memcmp(ring0.data(), h_rings + 5, ring0.size()) != 0;
                if (nc > 0) bad2 += std::memcmp(xm0.data(), h_x + (size_t) 5 * MT * H, xm0.size()) != 0;
                bad2 += h_words[0] != w0;
                cudaFree(d_pub);
                std::snprintf(what, sizeof what, "split route T=%d", T);
                report(what, l, bad2, (size_t) 6);
            }

            // tail: new = grouped into parts by the device plan, then dec_moe_combine reading the tiers' rows
            ck(cudaMemsetAsync(d_parts, 0xff, (size_t) K::kDmCap * H * 4, s0), "memset");
            K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, n, n,
                                     d_xq2, dv[0].scr, d_parts, s0);
            h_words[48] = 6;
            K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo, H, KK, T, h_words + 48, 6, s0);
            // old: the mapped rows (CPU rows, zeros at hits), copied with the hit rows zeroed, the hits added, then
            // the native combine
            for (int i = 0; i < n; ++i)
                std::memcpy(h_ymiss + (size_t) i * H, kind[i] == 0 ? std::vector<float>(H, 0.f).data()
                                                      : cpu.data() + (size_t) i * H, H * 4);
            float* d_parts2 = d_hit + 0;   // reuse: hit rows go to d_hit, parts2 is a second buffer below
            float* d_p2 = dmal<float>((size_t) K::kDmCap * H);
            ck(cudaMemsetAsync(d_hit, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
            K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, n, n,
                                     d_xq2, dv[0].scr, d_parts2, s0);
            K::copy_rows_from_mapped(d_p2, h_ymiss, n, H, d_plan->dst, d_plan->counts + 1, s0);
            K::moe_hit_add(d_p2, d_parts2, d_plan->dst, d_plan->counts + 1, n, H, s0);
            K::native_moe_combine_multi(d_p2, d_w, d_sh, d_bo2, H, KK, T, s0);
            ck(cudaStreamSynchronize(s0), "tail");
            const auto a = down(d_bo, (size_t) T * H), b = down(d_bo2, (size_t) T * H);
            size_t nd2 = 0;
            for (size_t i = 0; i < a.size(); ++i) nd2 += std::memcmp(&a[i], &b[i], 4) != 0;
            std::snprintf(what, sizeof what, "tail T=%d", T);
            report(what, l, nd2, a.size());
            cudaFree(d_p2);
            if (bench) {   // the combine alone, tiers landed 
                const double uc = time_us(s0, 200, [&] {
                    K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo, H, KK, T, h_words + 48, 6, s0);
                });
                std::printf("  bench: combine T=%d (%d cpu rows of %d): %.2f us per launch\n", T, nc, n, uc);
            }
            {   // the combine without its own wait (verify's call; DEC_GLUE & 4 takes the four-column kernel)
                float* d_bo3 = dmal<float>((size_t) MT * H);
                K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo3, H, KK, T, nullptr, 6, s0);
                ck(cudaStreamSynchronize(s0), "combine");
                const auto c3 = down(d_bo3, (size_t) T * H);
                size_t nd3 = 0;
                for (size_t i = 0; i < a.size(); ++i) nd3 += std::memcmp(&a[i], &c3[i], 4) != 0;
                std::snprintf(what, sizeof what, "combine, no wait T=%d", T);
                report(what, l, nd3, a.size());
                if (bench) {
                    const double uc = time_us(s0, 200, [&] {
                        K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo3, H, KK, T, nullptr, 6, s0);
                    });
                    std::printf("  bench: combine, no wait T=%d (%d cpu rows of %d): %.2f us per launch\n", T, nc, n, uc);
                }
                cudaFree(d_bo3);
            }
            // the combine waiting for the CPU pool's word itself (verify's call without --wait-stats, no wait launch):
            // the same sums with the word there from the start, and with the pool's word landing while it waits;
            // cpu_seen holds the ring
            for (int late = 0; late < 2; ++late) {
                float* d_bo4 = dmal<float>((size_t) MT * H);
                unsigned* d_seen = dmal<unsigned>(1);
                const unsigned prev = 5;
                ck(cudaMemcpy(d_seen, &prev, 4, cudaMemcpyHostToDevice), "seen");
                *(volatile uint32_t*) (h_words + 48) = late ? 5u : 6u;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo4, H, KK, T, nullptr, 6, s0, false, h_words + 48,
                                   d_seen);
                if (late) {   // the pool's word lands
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    *(volatile uint32_t*) (h_words + 48) = 6u;
                }
                ck(cudaStreamSynchronize(s0), "combine, own wait");
                const auto c4 = down(d_bo4, (size_t) T * H);
                size_t nd4 = 0;
                for (size_t i = 0; i < a.size(); ++i) nd4 += std::memcmp(&a[i], &c4[i], 4) != 0;
                const size_t nd_out = nd4;
                unsigned seen = 0;
                ck(cudaMemcpy(&seen, d_seen, 4, cudaMemcpyDeviceToHost), "seen");
                nd4 += seen != 6u;
                if (nd4 != 0)
                    std::printf("  combine, own wait%s T=%d: %zu output values differ, cpu_seen %u\n",
                                late ? " (landing late)" : "", T, nd_out, seen);
                std::snprintf(what, sizeof what, "combine, own wait%s T=%d", late ? " (landing late)" : "", T);
                report(what, l, nd4, a.size());
                if (bench && !late) {
                    cudaEvent_t b0, b1;
                    cudaEventCreate(&b0);
                    cudaEventCreate(&b1);
                    float best = 1e30f;
                    for (int rep = 0; rep < 25; ++rep) {
                        ck(cudaMemcpy(d_seen, &prev, 4, cudaMemcpyHostToDevice), "seen");
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        cudaEventRecord(b0, s0);
                        K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, d_sh, d_bo4, H, KK, T, nullptr, 6, s0, false,
                                           h_words + 48, d_seen);
                        cudaEventRecord(b1, s0);
                        cudaEventSynchronize(b1);
                        float ms = 0.f;
                        cudaEventElapsedTime(&ms, b0, b1);
                        if (rep > 4) best = std::min(best, ms);
                    }
                    std::printf("  bench: combine, own wait T=%d (%d cpu rows of %d): %.2f us per launch\n", T, nc, n,
                                1e3 * best);
                    cudaEventDestroy(b0);
                    cudaEventDestroy(b1);
                }
                cudaFree(d_bo4);
                cudaFree(d_seen);
            }
            {   // the shared expert's gate applied in the combine (each variant) against the same combine of a shared
                // row gated before it (one IEEE product per value, as the shared expert's own gate kernel makes it)
                std::vector<float> g(T), shg(sh.size());
                for (auto& v : g) v = 1.0f / (1.0f + std::exp(-nd(rng)));
                for (int t = 0; t < T; ++t)
                    for (int c = 0; c < (int) H; ++c) {
                        volatile float p = sh[(size_t) t * H + c] * g[t];
                        shg[(size_t) t * H + c] = p;
                    }
                float* d_g = dmal<float>(MT);
                float* d_shg = dmal<float>((size_t) MT * H);
                float* d_b0 = dmal<float>((size_t) MT * H);
                float* d_b1 = dmal<float>((size_t) MT * H);
                unsigned* d_seen = dmal<unsigned>(1);
                ck(cudaMemcpy(d_g, g.data(), (size_t) T * 4, cudaMemcpyHostToDevice), "g");
                ck(cudaMemcpy(d_shg, shg.data(), shg.size() * 4, cudaMemcpyHostToDevice), "shg");
                for (int var = 0; var < 3; ++var) {   // 0: no wait, 1: its own wait, 2: every block waits (host flag)
                    for (int gated = 0; gated < 2; ++gated) {
                        const unsigned prev = 5;
                        ck(cudaMemcpy(d_seen, &prev, 4, cudaMemcpyHostToDevice), "seen");
                        *(volatile uint32_t*) (h_words + 48) = 6u;
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        float* const out = gated ? d_b1 : d_b0;
                        const float* const shr = gated ? d_sh : d_shg;
                        const float* const gp = gated ? d_g : nullptr;
                        if (var == 0)
                            K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, shr, out, H, KK, T, nullptr, 6, s0, false, nullptr,
                                               nullptr, gp);
                        else if (var == 1)
                            K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, shr, out, H, KK, T, nullptr, 6, s0, false,
                                               h_words + 48, d_seen, gp);
                        else
                            K::dec_moe_combine(d_parts, h_cpu, d_plan, d_w, shr, out, H, KK, T, h_words + 48, 6, s0, false,
                                               nullptr, nullptr, gp);
                        ck(cudaStreamSynchronize(s0), "gated combine");
                    }
                    const auto c0 = down(d_b0, (size_t) T * H), c1 = down(d_b1, (size_t) T * H);
                    size_t ndg = 0;
                    for (size_t i = 0; i < c0.size(); ++i) ndg += std::memcmp(&c0[i], &c1[i], 4) != 0;
                    std::snprintf(what, sizeof what, "combine, gate inside (%s) T=%d",
                                  var == 0 ? "no wait" : var == 1 ? "own wait" : "block waits", T);
                    report(what, l, ndg, c0.size());
                }
                cudaFree(d_g);
                cudaFree(d_shg);
                cudaFree(d_b0);
                cudaFree(d_b1);
                cudaFree(d_seen);
            }

            // DEC_GLUE: the plan built in the gate/up launch (+ the ring launch's kinds) vs the route's plan,
            // its rows vs native_expert_grouped's (d_parts2); needs DEC_GU2=2
            if (K::dec_moe_glue_ok(L)) {
                std::vector<unsigned long long> pt(NE);
                for (int e = 0; e < NE; ++e)
                    pt[e] = res0[e] >= 0 ? (unsigned long long) (dv[0].blob + (size_t) res0[e] * L.bytes) : 0ull;
                unsigned long long* d_pt = dmal<unsigned long long>(NE);
                ck(cudaMemcpy(d_pt, pt.data(), NE * 8, cudaMemcpyHostToDevice), "ptab");
                float* d_p4 = dmal<float>((size_t) K::kDmCap * H);
                void* d_pub = dmal<uint8_t>(xq_stride);
                ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                K::DmRouteArgs r2 = ra;
                r2.mode = 2; r2.xq_pub = d_pub; r2.kinds = true;
                K::dec_moe_route(r2, s0);
                K::DmGlueArgs ga;
                ga.ids = d_ids; ga.ptab = d_pt; ga.n_tok = T; ga.k = (int) KK; ga.n_expert = (int) NE;
                ga.plan = d_plan; ga.xq = d_xq2;
                K::dec_moe_glue(L, ga, dv[0].scr, d_p4, n, s0);
                ck(cudaStreamSynchronize(s0), "glue");
                K::DmPlan P2;
                ck(cudaMemcpy(&P2, d_plan, sizeof P2, cudaMemcpyDeviceToHost), "plan");
                size_t bad3 = std::memcmp(P.counts, P2.counts, sizeof P.counts) != 0;
                for (int g = 0; g <= P.counts[0]; ++g) bad3 += P.start[g] != P2.start[g] || (g < P.counts[0] && P.ptr[g] != P2.ptr[g]);
                for (int e = 0; e < P.counts[1]; ++e) bad3 += P.dst[e] != P2.dst[e] || P.tok[e] != P2.tok[e];
                for (int i = 0; i < n; ++i) bad3 += P.kind[i] != P2.kind[i];
                std::snprintf(what, sizeof what, "glue plan T=%d", T);
                report(what, l, bad3, (size_t) n);
                const auto a4 = down(d_parts2, (size_t) n * H), b4 = down(d_p4, (size_t) n * H);
                std::vector<int> rows;
                for (int i = 0; i < n; ++i) if (kind[i] == 0) rows.push_back(i);
                std::snprintf(what, sizeof what, "glue rows T=%d (%d groups)", T, P.counts[0]);
                report(what, l, ndiff_rows(a4, b4, rows), rows.size() * H);
                {   // the ring's work in the launch: the ring, the mapped rows, the words and the kinds as route's
                    const std::vector<uint8_t> ring0((const uint8_t*) (h_rings + 5), (const uint8_t*) (h_rings + 6));
                    const std::vector<uint8_t> xm0((const uint8_t*) (h_x + (size_t) 5 * MT * H),
                                                   (const uint8_t*) (h_x + (size_t) 5 * MT * H + (size_t) T * H));
                    const size_t nq = (size_t) T * H / 32 * 36;
                    h_rings[5] = K::DmRing{};
                    std::memset(h_x + (size_t) 5 * MT * H, 0, (size_t) T * H * 4);
                    std::memset(h_xq + 5 * xq_stride, 0, nq);
                    h_words[0] = 0;
                    ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                    ck(cudaMemsetAsync(d_p4, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                    K::DmGlueArgs gr = ga;
                    gr.ring_on = true; gr.ring = r2;
                    K::dec_moe_glue(L, gr, dv[0].scr, d_p4, n, s0);
                    ck(cudaStreamSynchronize(s0), "glue ring");
                    K::DmPlan P3;
                    ck(cudaMemcpy(&P3, d_plan, sizeof P3, cudaMemcpyDeviceToHost), "plan");
                    size_t bad4 = std::memcmp(P.counts, P3.counts, sizeof P.counts) != 0;
                    for (int g = 0; g <= P.counts[0]; ++g) bad4 += P.start[g] != P3.start[g] || (g < P.counts[0] && P.ptr[g] != P3.ptr[g]);
                    for (int e = 0; e < P.counts[1]; ++e) bad4 += P.dst[e] != P3.dst[e] || P.tok[e] != P3.tok[e];
                    for (int i = 0; i < n; ++i) bad4 += P.kind[i] != P3.kind[i];
                    bad4 += std::memcmp(ring0.data(), h_rings + 5, ring0.size()) != 0;
                    if (nc > 0) bad4 += std::memcmp(xm0.data(), h_x + (size_t) 5 * MT * H, xm0.size()) != 0;
                    bad4 += h_words[0] != 6;
                    std::snprintf(what, sizeof what, "glue ring T=%d", T);
                    report(what, l, bad4, (size_t) n);
                    const auto c4 = down(d_p4, (size_t) n * H);
                    std::snprintf(what, sizeof what, "glue ring rows T=%d", T);
                    report(what, l, ndiff_rows(a4, c4, rows), rows.size() * H);
                    // the ring's block started with the read and released by its counts (verify's call): the same header,
                    // kinds, words and q8_1 rows, every token's FFN input rows, the counts zeroed
                    {
                        std::memset(h_xq + 5 * xq_stride, 0, nq);
                        std::memset(h_x + (size_t) 5 * MT * H, 0, (size_t) T * H * 4);
                        h_rings[5] = K::DmRing{};
                        h_words[0] = 0;
                        unsigned int* d_pc = dmal<unsigned int>(3);
                        const unsigned int counts[3] = {81, 1, 1};
                        ck(cudaMemcpy(d_pc, counts, 12, cudaMemcpyHostToDevice), "counts");
                        ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                        ck(cudaMemsetAsync(d_p4, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                        K::DmGlueArgs gp = ga;
                        gp.ring_on = true; gp.ring = r2; gp.ring.pub_ctr = d_pc; gp.ring.pub_target = counts[0];
                        K::dec_moe_glue(L, gp, dv[0].scr, d_p4, n, s0);
                        ck(cudaStreamSynchronize(s0), "glue ring, counted");
                        K::DmPlan P5;
                        ck(cudaMemcpy(&P5, d_plan, sizeof P5, cudaMemcpyDeviceToHost), "plan");
                        unsigned int left[3] = {1, 1, 1};
                        ck(cudaMemcpy(left, d_pc, 12, cudaMemcpyDeviceToHost), "counts");
                        size_t bad5 = std::memcmp(P.counts, P5.counts, sizeof P.counts) != 0 || left[0] != 0 || left[1] != 0 ||
                                      left[2] != 0;
                        for (int g = 0; g <= P.counts[0]; ++g) bad5 += P.start[g] != P5.start[g] || (g < P.counts[0] && P.ptr[g] != P5.ptr[g]);
                        for (int e = 0; e < P.counts[1]; ++e) bad5 += P.dst[e] != P5.dst[e] || P.tok[e] != P5.tok[e];
                        for (int i = 0; i < n; ++i) bad5 += P.kind[i] != P5.kind[i];
                        bad5 += std::memcmp(ring0.data(), h_rings + 5, ring0.size()) != 0;
                        bad5 += std::memcmp(q1.data(), h_xq + 5 * xq_stride, nq) != 0;
                        for (int t = 0; t < T; ++t) {   // every token's row is the FFN input
                            const float* row = h_x + (size_t) 5 * MT * H + (size_t) t * H;
                            bad5 += std::memcmp(row, x.data() + (size_t) t * H, H * 4) != 0;
                        }
                        bad5 += h_words[0] != 6;
                        std::snprintf(what, sizeof what, "glue ring, counted T=%d", T);
                        report(what, l, bad5, (size_t) n);
                        const auto c5 = down(d_p4, (size_t) n * H);
                        std::snprintf(what, sizeof what, "glue ring rows, counted T=%d", T);
                        report(what, l, ndiff_rows(a4, c5, rows), rows.size() * H);
                        // the landed words (an adaptive round's copies into the slots): with the layer's word at the
                        // need the rows are the launch's without the words; with the word short, the blobs are garbage
                        // until a copy on another stream puts them back and raises the word after them, and the launch
                        // must still be waiting when that starts and end with the same rows
                        {
                            unsigned int* d_lw = dmal<unsigned int>(2);
                            uint32_t* h_lw = nullptr;
                            uint8_t* h_blobs = nullptr;
                            ck(cudaHostAlloc((void**) &h_lw, 64, cudaHostAllocPortable), "pinned");
                            ck(cudaHostAlloc((void**) &h_blobs, blobs.size(), cudaHostAllocPortable), "pinned");
                            std::memcpy(h_blobs, blobs.data(), blobs.size());
                            h_lw[0] = 1u;
                            cudaStream_t sc;
                            ck(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking), "stream");
                            cudaEvent_t e_glue, e_word;
                            ck(cudaEventCreate(&e_glue), "event");
                            ck(cudaEventCreate(&e_word), "event");
                            for (int counted = 0; counted < 2; ++counted) {
                                for (int late = 0; late < 2; ++late) {
                                    K::DmGlueArgs gl = counted ? gp : gr;
                                    gl.landed = d_lw;
                                    gl.landed_need = d_lw + 1;
                                    ck(cudaMemset(d_lw, 0, 8), "words");
                                    if (late) {
                                        ck(cudaMemcpy(d_lw + 1, h_lw, 4, cudaMemcpyHostToDevice), "need");
                                        ck(cudaMemset(dv[0].blob, 0x5a, blobs.size()), "garble");
                                    }
                                    ck(cudaMemcpy(d_pc, counts, 12, cudaMemcpyHostToDevice), "counts");
                                    ck(cudaMemsetAsync(d_p4, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                                    ck(cudaDeviceSynchronize(), "landed setup");
                                    K::dec_moe_glue(L, gl, dv[0].scr, d_p4, n, s0);
                                    ck(cudaEventRecord(e_glue, s0), "event");
                                    bool early = false;
                                    if (late) {
                                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                        early = cudaEventQuery(e_glue) == cudaSuccess;
                                        ck(cudaMemcpyAsync(dv[0].blob, h_blobs, blobs.size(), cudaMemcpyHostToDevice, sc), "blobs back");
                                        ck(cudaMemcpyAsync(d_lw, h_lw, 4, cudaMemcpyHostToDevice, sc), "word");
                                        ck(cudaEventRecord(e_word, sc), "event");
                                    }
                                    ck(cudaStreamSynchronize(s0), "landed glue");
                                    ck(cudaStreamSynchronize(sc), "landed copies");
                                    float after = 1.0f;
                                    if (late) ck(cudaEventElapsedTime(&after, e_word, e_glue), "elapsed");
                                    const auto c6 = down(d_p4, (size_t) n * H);
                                    std::snprintf(what, sizeof what, "glue rows, landed%s%s T=%d", counted ? ", counted" : "",
                                                  late ? " late" : "", T);
                                    report(what, l, ndiff_rows(a4, c6, rows) + (early ? 1 : 0) + (after < 0.0f ? 1 : 0),
                                           rows.size() * H);
                                }
                            }
                            cudaEventDestroy(e_glue);
                            cudaEventDestroy(e_word);
                            cudaStreamDestroy(sc);
                            cudaFreeHost(h_blobs);
                            cudaFreeHost(h_lw);
                            cudaFree(d_lw);
                        }
                        cudaFree(d_pc);
                    }
                    std::memcpy(h_x + (size_t) 5 * MT * H, xm0.data(), xm0.size());
                }
                if (bench) {
                    const double ug = time_us(s0, 200, [&] { K::dec_moe_glue(L, ga, dv[0].scr, d_p4, n, s0); });
                    const double ur = time_us(s0, 200, [&] {
                        K::DmRouteArgs r1 = ra;
                        r1.mode = 1;
                        K::dec_moe_route(r1, s0);
                        K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                                 n, n, d_xq2, dv[0].scr, d_parts2, s0);
                    });
                    unsigned int* d_lw0 = dmal<unsigned int>(2);   // landed words at their need: the check alone
                    K::DmGlueArgs gw = ga;
                    gw.landed = d_lw0;
                    gw.landed_need = d_lw0 + 1;
                    const double ul = time_us(s0, 200, [&] { K::dec_moe_glue(L, gw, dv[0].scr, d_p4, n, s0); });
                    cudaFree(d_lw0);
                    const double mb = (double) P.counts[0] * (double) L.bytes / 1e6;   // the VRAM groups' expert bytes
                    std::printf("  bench: T=%d route + grouped %.2f us, glue %.2f us, glue with landed words %.2f us (%d groups, "
                                "%.1f MB: %.0f vs %.0f GB/s)\n",
                                T, ur, ug, ul, P.counts[0], mb, mb * 1e3 / ur, mb * 1e3 / ug);
                }
                cudaFree(d_pt); cudaFree(d_p4); cudaFree(d_pub);
            } else {
                std::printf("  glue: skipped (DEC_GU2=2 not set or formats)\n");
            }

            // the staged share: the last stage_k[m] of a layer's m distinct CPU misses go to CUDA0, their blobs
            // copied from host RAM into staging (dec_moe_stage) and computed by its grouped kernel.  The route's, the
            // ring's, the glue's and the stage kernel's kinds against a CPU reference, the staged lists and bytes, and
            // the tail against the same window with those experts resident in VRAM (the same bits)
            if (L.bytes % 16 == 0) {
                const int64_t stride = K::dec_moe_stage_stride((int64_t) L.bytes);
                // host copies of the blobs, each 16-byte aligned at a varying offset within its 128-byte line
                const size_t hstride = ((size_t) L.bytes + 255) / 128 * 128;
                uint8_t* h_sb = hmap<uint8_t>((size_t) NX * hstride);
                for (int x2 = 0; x2 < NX; ++x2)
                    std::memcpy(h_sb + (size_t) x2 * hstride + (size_t) (x2 % 8) * 16, blobs.data() + (size_t) x2 * L.bytes,
                                L.bytes);
                uint8_t* d_alias = nullptr;
                ck(cudaHostGetDevicePointer((void**) &d_alias, h_sb, 0), "host alias");
                std::vector<unsigned long long> src(NE, 0ull);
                for (int e = 0; e < NE; ++e)   // a few experts have no host address: never staged
                    if (e % 11 != 7)
                        src[e] = (unsigned long long) (d_alias + (size_t) (e % NX) * hstride + (size_t) ((e % NX) % 8) * 16);
                unsigned long long* d_src = dmal<unsigned long long>(NE);
                ck(cudaMemcpy(d_src, src.data(), NE * 8, cudaMemcpyHostToDevice), "stage src");
                std::vector<int32_t> kt(K::kDmCap + 1, 0);
                for (int m = 1; m <= K::kDmCap; ++m) kt[m] = std::min(K::kDmStageMax, (2 * m + 2) / 5);
                int32_t* d_kt = dmal<int32_t>(K::kDmCap + 1);
                ck(cudaMemcpy(d_kt, kt.data(), kt.size() * 4, cudaMemcpyHostToDevice), "stage k");
                uint8_t* d_stg = dmal<uint8_t>((size_t) K::kDmStageMax * stride);
                // the reference: CUDA0's kinds, then the last stage_k[m] distinct misses with a host address, every
                // entry of theirs
                std::vector<int32_t> rk(n);
                for (int i = 0; i < n; ++i) rk[i] = res0[ids[i]] >= 0 ? 0 : -1;
                std::vector<int> firsts;
                for (int i = 0; i < n; ++i) {
                    bool first = rk[i] < 0;
                    for (int j = 0; first && j < i; ++j) first = !(ids[j] == ids[i] && rk[j] < 0);
                    if (first) firsts.push_back(i);
                }
                const int m = (int) firsts.size(), kk = kt[(size_t) m];
                std::vector<int> elig;
                for (int i : firsts) if (src[(size_t) ids[i]] != 0ull) elig.push_back(i);
                std::vector<int> staged;   // the staged experts
                for (size_t q = elig.size() > (size_t) kk ? elig.size() - (size_t) kk : 0; q < elig.size(); ++q)
                    staged.push_back(ids[elig[q]]);
                auto is_staged = [&](int e) { return std::find(staged.begin(), staged.end(), e) != staged.end(); };
                int rnc = 0;
                for (int i = 0; i < n; ++i) {
                    if (rk[i] < 0 && is_staged(ids[i])) rk[i] = 1;
                    rnc += rk[i] < 0;
                }
                uint32_t* sw = hmap<uint32_t>(64);
                K::DmRouteArgs rs = ra;
                rs.host_seq_m = sw;
                rs.stage_src = d_src; rs.stage_k = d_kt;
                ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                std::memset((void*) (h_rings + 5), 0, sizeof(K::DmRing));
                K::dec_moe_route(rs, s0);
                K::dec_moe_stage(rs, d_stg, stride, (int64_t) L.bytes, 8, s0);
                ck(cudaStreamSynchronize(s0), "staged route");
                K::DmPlan S;
                ck(cudaMemcpy(&S, d_plan, sizeof S, cudaMemcpyDeviceToHost), "plan");
                size_t bst = 0;
                for (int i = 0; i < n; ++i) bst += S.kind[i] != rk[i] || h_rings[5].kind[i] != rk[i];
                bst += h_rings[5].n_cpu != rnc || S.counts[2] != rnc || S.scounts[0] != (int) staged.size();
                const auto stg = down(d_stg, (size_t) K::kDmStageMax * stride);
                std::vector<int> sseen(n, 0);
                int nse = 0;
                for (int g = 0; g < S.scounts[0] && g < K::kDmStageMax; ++g) {
                    const int e0 = S.sstart[g], e1 = S.sstart[g + 1];
                    const int ex = e1 > e0 && S.sdst[e0] >= 0 && S.sdst[e0] < n ? ids[S.sdst[e0]] : -1;
                    bst += ex < 0 || !is_staged(ex);
                    for (int e = e0; e < e1; ++e) {
                        const int i = S.sdst[e];
                        bst += i < 0 || i >= n || (i < n && ids[i] != ex) || S.stok[e] != i / KK;
                        if (i >= 0 && i < n) ++sseen[i];
                        ++nse;
                    }
                    const unsigned long long off = S.sptr[g] - (unsigned long long) (d_stg + (size_t) g * stride);
                    bst += ex < 0 || off != (src[(size_t) ex] & 127ull);
                    if (ex >= 0 && off < 128)
                        bst += std::memcmp(stg.data() + (size_t) g * stride + off, blobs.data() + (size_t) (ex % NX) * L.bytes,
                                           L.bytes) != 0;
                }
                bst += S.scounts[1] != nse;
                for (int i = 0; i < n; ++i) bst += (rk[i] == 1) != (sseen[i] == 1);
                std::snprintf(what, sizeof what, "staged split T=%d (%d of %d misses)", T, (int) staged.size(), m);
                report(what, l, bst, (size_t) n);
                // the tail: CUDA0's groups and the staged ones into parts, then the combine, against the window with the
                // staged experts resident at the slots of their blobs
                std::vector<int32_t> res2 = res0;
                for (int e : staged) res2[(size_t) e] = e % NX;
                int32_t* d_res2 = dmal<int32_t>(NE);
                ck(cudaMemcpy(d_res2, res2.data(), NE * 4, cudaMemcpyHostToDevice), "res2");
                K::DmPlan* d_plan2 = dmal<K::DmPlan>(1);
                float* d_pa = dmal<float>((size_t) K::kDmCap * H);
                float* d_pb = dmal<float>((size_t) K::kDmCap * H);
                float* d_ba = dmal<float>((size_t) MT * H);
                float* d_bb = dmal<float>((size_t) MT * H);
                K::DmRouteArgs rr = ra;
                rr.res0 = d_res2; rr.plan = d_plan2;
                rr.ring_m = h_rings + 7; rr.x_m = h_x + (size_t) 7 * MT * H; rr.xq_m = h_xq + 7 * xq_stride;
                rr.host_seq_m = sw + 32;
                K::dec_moe_route(rr, s0);
                ck(cudaMemsetAsync(d_pa, 0xff, (size_t) K::kDmCap * H * 4, s0), "memset");
                ck(cudaMemsetAsync(d_pb, 0xff, (size_t) K::kDmCap * H * 4, s0), "memset");
                K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok, n, n,
                                         d_xq2, dv[0].scr, d_pa, s0);
                K::native_expert_grouped(L, d_plan->sptr, d_plan->sstart, d_plan->scounts, d_plan->sdst, d_plan->stok, n,
                                         n, d_xq2, dv[0].scr, d_pa, s0, 4);
                K::native_expert_grouped(L, d_plan2->ptr, d_plan2->start, d_plan2->counts, d_plan2->dst, d_plan2->tok, n,
                                         n, d_xq2, dv[0].scr, d_pb, s0);
                h_words[48] = 6;
                K::dec_moe_combine(d_pa, h_cpu, d_plan, d_w, d_sh, d_ba, H, KK, T, h_words + 48, 6, s0);
                K::dec_moe_combine(d_pb, h_cpu, d_plan2, d_w, d_sh, d_bb, H, KK, T, h_words + 48, 6, s0);
                ck(cudaStreamSynchronize(s0), "staged tail");
                std::vector<int> srows;
                for (int i = 0; i < n; ++i) if (rk[i] == 1) srows.push_back(i);
                const auto pa = down(d_pa, (size_t) n * H), pb = down(d_pb, (size_t) n * H);
                std::snprintf(what, sizeof what, "staged rows T=%d", T);
                report(what, l, ndiff_rows(pa, pb, srows), srows.size() * H);
                const auto ba = down(d_ba, (size_t) T * H), bb = down(d_bb, (size_t) T * H);
                size_t ndt = 0;
                for (size_t i = 0; i < ba.size(); ++i) ndt += std::memcmp(&ba[i], &bb[i], 4) != 0;
                std::snprintf(what, sizeof what, "staged tail T=%d", T);
                report(what, l, ndt, ba.size());
                if (K::dec_moe_glue_ok(L)) {   // the ring's kinds made inside the glue launch split the same way
                    std::vector<unsigned long long> pt(NE);
                    for (int e = 0; e < NE; ++e)
                        pt[(size_t) e] = res0[(size_t) e] >= 0 ? (unsigned long long) (dv[0].blob + (size_t) res0[(size_t) e] * L.bytes)
                                                              : 0ull;
                    unsigned long long* d_pt = dmal<unsigned long long>(NE);
                    ck(cudaMemcpy(d_pt, pt.data(), NE * 8, cudaMemcpyHostToDevice), "ptab");
                    void* d_pub = dmal<uint8_t>(xq_stride);
                    float* d_p4 = dmal<float>((size_t) K::kDmCap * H);
                    std::memset((void*) (h_rings + 5), 0, sizeof(K::DmRing));
                    ck(cudaMemsetAsync(d_plan, 0xee, sizeof(K::DmPlan), s0), "memset");
                    K::DmRouteArgs r2 = rs;
                    r2.mode = 2; r2.xq_pub = d_pub; r2.kinds = true;
                    K::DmGlueArgs gs;
                    gs.ids = d_ids; gs.ptab = d_pt; gs.n_tok = T; gs.k = (int) KK; gs.n_expert = (int) NE;
                    gs.plan = d_plan; gs.xq = d_xq2; gs.ring_on = true; gs.ring = r2;
                    K::dec_moe_glue(L, gs, dv[0].scr, d_p4, n, s0);
                    ck(cudaStreamSynchronize(s0), "staged glue");
                    K::DmPlan G2;
                    ck(cudaMemcpy(&G2, d_plan, sizeof G2, cudaMemcpyDeviceToHost), "plan");
                    size_t badg = 0;
                    for (int i = 0; i < n; ++i) badg += G2.kind[i] != rk[i] || h_rings[5].kind[i] != rk[i];
                    badg += h_rings[5].n_cpu != rnc;
                    std::snprintf(what, sizeof what, "staged glue kinds T=%d", T);
                    report(what, l, badg, (size_t) n);
                    cudaFree(d_pt); cudaFree(d_pub); cudaFree(d_p4);
                }
                if (bench && !staged.empty())   // the copy over the link alone, by the blocks it launches
                    for (int blocks : {8, 16, 32, 64}) {
                        const double us = time_us(s0, 100, [&] {
                            K::dec_moe_stage(rs, d_stg, stride, (int64_t) L.bytes, blocks, s0);
                        });
                        const double mb = (double) staged.size() * (double) L.bytes / 1e6;
                        std::printf("  bench: staged copy T=%d, %d blobs (%.1f MB), %d blocks: %.1f us, %.1f GB/s\n", T,
                                    (int) staged.size(), mb, blocks, us, mb * 1e-3 / (us * 1e-6));
                    }
                if (bench && !staged.empty())   // the staged groups' compute, by the block rows it launches
                    for (int rows : {4, 16}) {
                        const double us = time_us(s0, 100, [&] {
                            K::native_expert_grouped(L, d_plan->sptr, d_plan->sstart, d_plan->scounts, d_plan->sdst,
                                                     d_plan->stok, n, n, d_xq2, dv[0].scr, d_pa, s0, rows);
                        });
                        std::printf("  bench: staged compute T=%d, %d groups, %d block rows: %.1f us\n", T, (int) staged.size(),
                                    rows, us);
                    }
                for (void* q : {(void*) d_src, (void*) d_kt, (void*) d_stg, (void*) d_res2, (void*) d_plan2, (void*) d_pa,
                                (void*) d_pb, (void*) d_ba, (void*) d_bb})
                    cudaFree(q);
                cudaFreeHost(h_sb);
                cudaFreeHost(sw);
                ck(cudaMemcpy(d_plan, &P, sizeof P, cudaMemcpyHostToDevice), "plan back");
            }

            // the fused VRAM call vs native_expert_grouped (d_parts2 holds the grouped rows)
            {
                float* d_p3 = dmal<float>((size_t) K::kDmCap * H);
                ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_p3, d_vsync, s0);
                ck(cudaStreamSynchronize(s0), "vram");
                const auto a3 = down(d_parts2, (size_t) n * H), b3 = down(d_p3, (size_t) n * H);
                std::vector<int> rows;
                for (int i = 0; i < n; ++i) if (kind[i] == 0) rows.push_back(i);
                std::snprintf(what, sizeof what, "fused VRAM T=%d (%d groups)", T, P.counts[0]);
                report(what, l, ndiff_rows(a3, b3, rows), rows.size() * H);
                ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                int major = 0;
                ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), "compute capability");
                for (int kind : {1, 2, 3}) {
                    // the TMA kernel stages 8 gate/up or 32 down rows in 11776 bytes: wider rows have no such kernel;
                    // its bulk copies need compute capability 9.0
                    if (kind == 1 && (L.gu_row * 8 > 11776 || L.d_row * 32 > 11776 || major < 9)) continue;
                    ck(cudaMemsetAsync(d_p3, 0, (size_t) K::kDmCap * H * 4, s0), "memset");
                    K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_p3, d_vsync, s0, kind, n);
                    ck(cudaStreamSynchronize(s0), "vram kind");
                    const auto t3 = down(d_p3, (size_t) n * H);
                    std::snprintf(what, sizeof what, "VRAM kind %d T=%d (%d groups)", kind, T, P.counts[0]);
                    report(what, l, ndiff_rows(a3, t3, rows), rows.size() * H);
                }
                cudaFree(d_p3);
            }
            if (bench && T == 3) {
                // eight copies of the blobs (past the L2), the route kernel pointing each call at the next one
                uint8_t* big = dmal<uint8_t>(8 * blobs.size());
                for (int c = 0; c < 8; ++c)
                    ck(cudaMemcpy(big + c * blobs.size(), blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "big");
                int rot = 0;
                auto route = [&] {
                    K::DmRouteArgs rb = ra;
                    rb.cache_base = big + (size_t) (rot++ % 8) * blobs.size();
                    K::dec_moe_route(rb, s0);
                };
                const double r0 = time_us(s0, 200, route);
                const double ug = time_us(s0, 200, [&] {
                    route();
                    K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                             n, n, d_xq2, dv[0].scr, d_parts, s0);
                }) - r0;
                const double uf = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0); }) - r0;
                // the bulk-copy kernel stages 8 gate/up and 32 down rows: formats with wider rows are not timed (nan)
                int cc = 0;
                cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, 0);
                const bool tma_fits = L.gu_row * 8 <= 11776 && L.d_row * 32 <= 11776 && cc >= 9;
                const double ut = !tma_fits ? std::nan("") :
                    time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 1); }) - r0;
                const double u2 = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 2, n); }) - r0;
                const double u4 = time_us(s0, 200, [&] { route(); K::dec_moe_vram(L, d_plan, d_xq2, d_vscr, d_parts, d_vsync, s0, 3, n); }) - r0;
                auto phase = [&](int ph) {
                    K::native_expert_set_mode(-1, ph);
                    const double u = time_us(s0, 200, [&] {
                        route();
                        K::native_expert_grouped(L, d_plan->ptr, d_plan->start, d_plan->counts, d_plan->dst, d_plan->tok,
                                                 n, n, d_xq2, dv[0].scr, d_parts, s0);
                    }) - r0;
                    K::native_expert_set_mode(-1, 0);
                    return u;
                };
                const double ugu = phase(1), udn = phase(2);
                std::printf("  bench: grouped phases: gate/up+SwiGLU %.1f us (%.0f GB/s), down %.1f us (%.0f GB/s)\n", ugu,
                            P.counts[0] * (double) (2 * L.up_off) / ugu / 1e3, udn,
                            P.counts[0] * (double) (L.bytes - L.down_off) / udn / 1e3);
                const double gb = P.counts[0] * (double) L.bytes / 1e3;
                std::printf("  bench: VRAM share T=%d, %d groups (%d entries, %.1f MB), past the L2: route %.1f us | "
                            "grouped %.1f us %.0f GB/s | fused %.1f us %.0f GB/s | TMA %.1f us %.0f GB/s | PF1 %.1f us %.0f GB/s | "
                            "PF2 %.1f us %.0f GB/s\n",
                            T, P.counts[0], P.counts[1], gb / 1e3, r0, ug, gb / ug, uf, gb / uf, ut, gb / ut, u2, gb / u2, u4, gb / u4);
                cudaFree(big);
            }
        }

        for (int d = 0; d < (two ? 2 : 1); ++d) {
            ck(cudaSetDevice(d), "dev");
            ck(cudaDeviceSynchronize(), "sync");
            cudaFree(dv[d].blob);
        }
        ck(cudaSetDevice(0), "dev0");
    }
    std::printf("dec_moe_parity: %s\n", g_fail ? "FAILED" : "all bitwise equal");
    return g_fail ? 1 : 0;
}
