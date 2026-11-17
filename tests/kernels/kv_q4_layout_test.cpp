// tests/kernels/kv_q4_layout_test.cpp - the q4_0 KV block's rule on the CPU (halo/kernels/kv_q4_layout.hpp): the scale is
// the block's signed largest-|value| over -8, the codes are ggml's trunc(x / d + 8.5) clamped to 0..15, the stored code's
// value is (code - 8) * d, and the eight-code reader every kernel uses agrees with the element layout. The dense path's
// K8V4 and rotated 4-bit caches and the sparse path's q4_0 cache all quantize through this header, so a change here that
// moves a bit moves both paths' bytes.
//
// No GPU, no driver: a CPU test (`ctest -L cpu`).
#include "halo/kernels/kv_q4_layout.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace k = halo::kernels;

namespace {
int g_fail = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

float f16(uint16_t h) { return k::f32_from_f16(h); }

// The element layout written out: element j is the low nibble of qs[j] for j < 16 and the high nibble of qs[j - 16]
// from 16 on.  The oracle the kernel-facing reader is checked against.
int element_code(const k::KvQ4Block& b, int j) {
    return j < 16 ? (b.qs[j] & 0x0F) : ((b.qs[j - 16] >> 4) & 0x0F);
}
}  // namespace

int main() {
    std::printf("kv_q4_layout: block %zu bytes, %d values, head bytes 256 -> %llu, 128 -> %llu\n", sizeof(k::KvQ4Block),
                k::kKvQ4Block, (unsigned long long) k::kv_q4_head_bytes(256), (unsigned long long) k::kv_q4_head_bytes(128));
    expect(sizeof(k::KvQ4Block) == 18, "a block is 18 bytes (an fp16 scale and 16 code bytes)");
    expect(k::kv_q4_head_bytes(256) == 8 * 18, "a 256-wide head row is eight blocks");
    expect(k::kv_q4_head_bytes(128) == 4 * 18, "a 128-wide head row is four blocks");

    std::mt19937 rng(20261117u);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---- the scale: the signed largest-|value| over -8, ties to the larger value; codes against the unrounded delta
    {
        float x[32];
        for (int round = 0; round < 200; ++round) {
            for (float& v : x) v = nd(rng) * 3.0f;
            const k::KvQ4Block b = k::kv_q4_quantize_host(x);
            float amax = 0.0f, mval = 0.0f;
            for (float v : x)
                if (std::fabs(v) > amax || (std::fabs(v) == amax && v > mval)) {
                    amax = std::fabs(v);
                    mval = v;
                }
            expect(b.d == k::kv_q4_scale_bits(mval), "the stored scale is f16(largest-|value| / -8)");
            const float d = k::kv_q4_delta(mval);
            const float ds = f16(b.d);
            for (int j = 0; j < 32; ++j) {
                expect(element_code(b, j) == k::kv_q4_code(x[j], d), "each code is trunc(x / d + 8.5) against the delta");
                // the stored (fp16) scale decides the value: half a step for a value inside the range, a whole step at
                // the clipped end (the largest positive value maps to code 0, and its negative twin to 15)
                // the codes are rounded against the delta, the value is (code - 8) times the STORED fp16 scale, so a
                // value's error is half a step plus the scale's own fp16 rounding (|x| * 2^-11 at worst)
                const float back = k::kv_q4_dequant(element_code(b, j), ds);
                const float slack = std::fabs(x[j]) * (1.0f / 1024.0f) + 1e-6f;
                expect(std::fabs(back - x[j]) <= (std::fabs(ds) + slack), "a value is within one step of its code's value");
                // inside the codes' span (|x / d| <= 7.5) the code is the nearest; past it the top code clips, and the
                // error is up to one step. That asymmetry is ggml's q4_0 (the delta is max / -8), not a bug here.
                if (std::fabs(x[j] / d) <= 7.5f)
                    expect(std::fabs(back - x[j]) <= 0.5f * std::fabs(ds) + slack, "an in-range value is within half a step");
            }
        }
    }

    // ---- ties resolve to the larger value, in any order, so a warp's shuffle reduction and a host loop agree
    {
        float a[32] = {0.0f}, b[32] = {0.0f};
        a[0] = 2.0f, a[5] = -2.0f;
        b[5] = -2.0f, b[31] = 2.0f;
        const k::KvQ4Block qa = k::kv_q4_quantize_host(a), qb = k::kv_q4_quantize_host(b);
        expect(qa.d == qb.d, "a tie on |x| picks one scale whatever the order");
        expect(f16(qa.d) == -0.25f, "the tie picks the larger value (+2 / -8), not the smaller");
        expect(element_code(qa, 0) == 0 && element_code(qa, 5) == 15, "the largest value is code 0 and its twin clips to 15");
        expect(k::kv_q4_dequant(0, f16(qa.d)) == 2.0f, "code 0 is the largest value exactly");
    }

    // ---- an all-zero block: a zero scale, every code 8, every value 0
    {
        float z[32] = {0.0f};
        const k::KvQ4Block b = k::kv_q4_quantize_host(z);
        expect(f16(b.d) == 0.0f, "an all-zero block stores a zero scale");
        for (int j = 0; j < 32; ++j) expect(element_code(b, j) == 8, "an all-zero block stores code 8");
        expect(k::kv_q4_dequant(8, f16(b.d)) == 0.0f, "code 8 at a zero scale is zero");
    }

    // ---- the eight-code reader: every dim of a 256-wide head row, read the way the decode and prompt kernels read it
    {
        float row[256];
        for (float& v : row) v = nd(rng) * 0.5f;
        k::KvQ4Block blocks[8];
        for (int b = 0; b < 8; ++b) blocks[b] = k::kv_q4_quantize_host(&row[b * 32]);
        int bad = 0;
        for (int d = 0; d < 256; ++d) {
            const int d8 = (d & 31) & ~7;      // the reader's granularity: eight dims from a multiple of 8
            int c[8];
            k::kv_q4_codes8(blocks[d >> 5].qs, d8, c);
            if (c[d & 7] != element_code(blocks[d >> 5], d & 31)) ++bad;
            if (k::kv_q4_dequant(c[d & 7], f16(blocks[d >> 5].d)) !=
                k::kv_q4_dequant(element_code(blocks[d >> 5], d & 31), f16(blocks[d >> 5].d)))
                ++bad;
        }
        expect(bad == 0, "kv_q4_codes8 reads every dim the same code and value the element layout does");
        // and a 16-dim piece of a row is the way the prompt kernel's tile copy reads it
        bad = 0;
        for (int c16 = 0; c16 < 16; ++c16) {
            const k::KvQ4Block& blk = blocks[c16 >> 1];
            int c[16];
            k::kv_q4_codes8(blk.qs, (c16 & 1) * 16, c);
            k::kv_q4_codes8(blk.qs, (c16 & 1) * 16 + 8, c + 8);
            for (int e = 0; e < 16; ++e)
                if (c[e] != element_code(blk, (c16 & 1) * 16 + e)) ++bad;
        }
        expect(bad == 0, "a 16-dim piece of a row reads the codes of its own dims");
    }

    // ---- the reconstruction bound over random rows, printed for the report
    {
        double worst = 0.0, worst_rel = 0.0;
        for (int round = 0; round < 100; ++round) {
            float row[256];
            for (float& v : row) v = nd(rng) * nd(rng);
            float amax = 0.0f;
            for (float v : row) amax = std::max(amax, std::fabs(v));
            for (int b = 0; b < 8; ++b) {
                const k::KvQ4Block blk = k::kv_q4_quantize_host(&row[b * 32]);
                const float d = f16(blk.d);
                for (int j = 0; j < 32; ++j) {
                    const double e = std::fabs(k::kv_q4_dequant(element_code(blk, j), d) - row[b * 32 + j]);
                    worst = std::max(worst, e);
                    if (row[b * 32 + j] != 0.0f) worst_rel = std::max(worst_rel, e / std::fabs(row[b * 32 + j]));
                    if (e > (double) std::fabs(d) + 1e-6) ++g_fail;
                }
                expect(std::fabs(d) <= amax / 8.0f * 1.001f + 1e-9f, "the step is at most an eighth of the row's largest |value|");
            }
        }
        std::printf("  worst reconstruction error %.6g (absolute), %.6g (relative)\n", worst, worst_rel);
    }

    std::printf("kv_q4_layout: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
