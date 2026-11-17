// tests/kernels/dense_kv_planner_test.cpp - the dense planner's numbers for the Qwen3.8-27B geometry: a KV format's exact
// bytes per cell, the 262,144-cell totals the engine printed at int8 and fp16 (evals/2026-11-17/halocline-dense-quality/
// fit/), and the D8 checks that K8V4's 184,320 cells fit the UD-Q4_K_XL card once the MTP drafter's KV is a ring, with
// the roadmap's ~4.6 GiB left for KV (docs/roadmap.md, D8). No device is touched: these are the planner's own host
// functions (`ctest -L cpu`).
#include "halo/dense/model.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace d = halo::dense;

namespace {
int g_fail = 0;
constexpr uint64_t kMib = 1ull << 20, kGib = 1ull << 30;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// the 27B's shape, as its pack's metadata carries it (src/models/qwen35/meta.hpp reads these keys) and BUILD_SPEC §2
// lists them: 64 layers, 16 of them gated attention (three Gated DeltaNet layers to one), hidden 5,120, FFN 17,408,
// 24 query / 4 KV heads at width 256, a 64-wide partial rope, 248,320 tokens.
d::Geometry qwen3827b() {
    d::Geometry g;
    g.n_layers = 64;
    g.n_embd = 5120;
    g.n_ff = 17408;
    g.n_vocab = 248320;
    g.n_head = 24;
    g.n_kv = 4;
    g.head_dim = 256;
    g.rope.rot = 64;
    g.rope.theta = 1e7f;
    g.gdn_state = 128;
    g.gdn_k_heads = 16;
    g.gdn_v_heads = 48;
    g.gdn_conv = 4;
    for (int il = 0; il < g.n_layers; ++il) g.kinds.push_back((il + 1) % 4 == 0 ? d::Mixer::Attention : d::Mixer::Gdn);
    return g;
}

// what the engine charges a session and the model's scratch: the adapter's own formula (qwen35/adapter.cpp), MiB
uint64_t need_mib(const d::Geometry& g, int64_t cells, d::KvFormat kv) {
    return (d::shared_bytes(g, cells, kv) + d::session_bytes(g, cells, /*mtp=*/true, kv)) / kMib;
}

// the same with the MTP drafter's KV as a 32K-cell ring (D8 item 2) instead of full context: its own layer stays fp16
uint64_t need_mib_drafter_ring(const d::Geometry& g, int64_t cells, d::KvFormat kv, int64_t ring = 32768) {
    const uint64_t full = d::session_bytes(g, cells, /*mtp=*/true, kv);
    const uint64_t drafter = (uint64_t) cells * d::kv_cell_bytes(g, d::KvFormat::Fp16);   // the drafter's own layer, fp16
    return (d::shared_bytes(g, cells, kv) + full - drafter + (uint64_t) ring * d::kv_cell_bytes(g, d::KvFormat::Fp16)) / kMib;
}

// the most cells the UD-Q4_K_XL plan takes: the adapter's own test (the need plus a 512 MiB reserve against the free
// bytes), searched. `ring`: the MTP drafter's KV as a 32K-cell ring (D8 item 2) instead of full context.
constexpr uint64_t kFreeMib = 7597;   // free after the UD-Q4_K_XL weights, vision packs, evals/.../fit/
int64_t max_cells(const d::Geometry& g, d::KvFormat kv, bool ring) {
    int64_t lo = 1024, hi = 400000;
    while (lo < hi) {
        const int64_t mid = (lo + hi + 1) / 2;
        const uint64_t need = ring ? need_mib_drafter_ring(g, mid, kv) : need_mib(g, mid, kv);
        if (need + 512 <= kFreeMib) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

}  // namespace

int main() {
    const d::Geometry g = qwen3827b();
    std::printf("dense KV planner: %d attention layers of %d KV heads %d wide\n", g.attn_layers(), g.n_kv, g.head_dim);
    // one cell of one attention layer: K and V of every KV head, the scales each format stores beside the codes
    const uint64_t fp16 = d::kv_cell_bytes(g, d::KvFormat::Fp16), int8 = d::kv_cell_bytes(g, d::KvFormat::Int8),
                   k8v4 = d::kv_cell_bytes(g, d::KvFormat::K8V4);
    std::printf("  a cell: fp16 %llu B, int8 %llu B, K8V4 %llu B\n", (unsigned long long) fp16, (unsigned long long) int8,
                (unsigned long long) k8v4);
    expect(fp16 == 4096, "fp16: 4 KV heads x 256 values x 2 x 2");
    expect(int8 == 2112, "int8: codes and 4 fp16 scales per 64 values, K and V");
    expect(k8v4 == 1632, "K8V4: int8 K (1,056) plus 8 fp16-scaled q4_0 blocks a KV head (576)");

    // ---- the recorded planner outputs at 262,144 cells must not move (the formats' exact bytes live in one function)
    const int64_t ctx = 262144;
    const uint64_t at_int8 = need_mib(g, ctx, d::KvFormat::Int8), at_fp16 = need_mib(g, ctx, d::KvFormat::Fp16);
    std::printf("  262,144 cells: int8 %llu MiB, fp16 %llu MiB (fit/: 11,830 and 19,766)\n", (unsigned long long) at_int8,
                (unsigned long long) at_fp16);
    expect(at_int8 == 11830, "the int8 total at 262,144 cells is the 11,830 MiB the engine printed");
    expect(at_fp16 == 19766, "the fp16 total at 262,144 cells is the 19,766 MiB the engine printed");

    // ---- D8: the KV a K8V4 cache takes at the roadmap's 184,320 cells, and the same bytes at 4-bit's 261,120
    const int64_t k8v4_cells = 184320, q4_cells = 261120;
    const uint64_t kv_k8v4 = (uint64_t) k8v4_cells * g.attn_layers() * k8v4;
    std::printf("  K8V4 at %lld cells: KV %llu MiB; the roadmap leaves ~4.6 GiB for KV\n", (long long) k8v4_cells,
                (unsigned long long) (kv_k8v4 / kMib));
    expect(kv_k8v4 <= (uint64_t) (4.6 * (double) kGib), "K8V4's 184,320 cells fit the ~4.6 GiB left for KV");
    // rotated 4-bit K and V is 1,152 B a cell, and the roadmap's two contexts spend the same KV bytes: 17/12 of the cells
    expect((uint64_t) q4_cells * g.attn_layers() * 1152 == kv_k8v4, "261,120 4-bit cells are the same KV bytes as 184,320 K8V4");

    // ---- the whole plan: UD-Q4_K_XL's 7,597 MiB free after its weights, --spec 8 (the MTP drafter), vision packs
    const uint64_t free_mib = 7597, reserve_mib = 512;   // the adapter adds 512 MiB on top of shared_bytes' own 512
    const uint64_t k8v4_ring = need_mib_drafter_ring(g, k8v4_cells, d::KvFormat::K8V4);
    const uint64_t k8v4_full = need_mib(g, k8v4_cells, d::KvFormat::K8V4);
    std::printf("  184,320 cells on UD-Q4_K_XL: %llu MiB with the drafter's KV at full context, %llu with its ring (+%llu "
                "reserve) against %llu free\n",
                (unsigned long long) k8v4_full, (unsigned long long) k8v4_ring, (unsigned long long) reserve_mib,
                (unsigned long long) free_mib);
    expect(k8v4_ring + reserve_mib <= free_mib, "184,320 K8V4 cells fit once the drafter's KV is a ring");
    expect(k8v4_full + reserve_mib > free_mib, "and not with the drafter's KV at full context: the ring is what buys it");

    // ---- the roadmap's figure at 184K assumes the ring; print the exact maxima for the report
    const int64_t fp16_full = max_cells(g, d::KvFormat::Fp16, false);
    const int64_t int8_full = max_cells(g, d::KvFormat::Int8, false), int8_ring = max_cells(g, d::KvFormat::Int8, true);
    const int64_t k8v4_full_max = max_cells(g, d::KvFormat::K8V4, false), k8v4_ring_max = max_cells(g, d::KvFormat::K8V4, true);
    std::printf("  max cells on UD-Q4_K_XL: fp16 %lld; int8 %lld (ring %lld); K8V4 %lld (ring %lld); the roadmap says ~184K "
                "for K8V4\n",
                (long long) fp16_full, (long long) int8_full, (long long) int8_ring, (long long) k8v4_full_max,
                (long long) k8v4_ring_max);
    expect(k8v4_ring_max >= k8v4_cells, "the roadmap's 184,320 K8V4 cells are inside the ring plan's maximum");
    expect(k8v4_ring_max < 186000, "and the maximum is the roadmap's ~184K, not a larger range quoted by mistake");
    expect(k8v4_full_max < k8v4_cells, "without the ring, 184,320 K8V4 cells do not fit: D8 item 2 is part of the 184K");

    std::printf("dense_kv_planner: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
