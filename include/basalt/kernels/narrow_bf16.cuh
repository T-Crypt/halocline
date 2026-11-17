// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// tests/kernels/narrow_bf16_parity.cu compares the two ways on every input.
//
// Halocline builds for 8.9 alone, so only the 8.9 forms below remain (the direct conversions need 10.0+).
// Compute capability 8.9 has the E4M3 conversion (cvt.rn.f16x2.e4m3x2) but neither E2M1 conversion nor mul.rn.bf16x2
// (9.0).  Its device pass decodes E2M1 by a byte permute from the eight magnitudes' bit patterns, the sign bit OR'd in
// (every E2M1 value is exact in f16 and bf16, so these are the instruction's bits), and multiplies by fma.rn.bf16x2 with a
// -0 addend: x * y + -0 is the exact product rounded once, as mul.rn rounds it (a zero product keeps its sign).
#pragma once

#include <cuda_runtime.h>


namespace basalt::kernels {

/// bf16x2 a * b, both halves rounded once (mul.rn.bf16x2, 9.0+; 8.9: fma.rn.bf16x2 with a -0 addend, the same rounding of
/// the same exact product)
__device__ __forceinline__ unsigned mul_bf16x2(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

/// byte i of w (two E2M1 codes, the low nibble the lower half) as a pair of 16-bit patterns: hi_* and lo_* are the high
/// and low bytes of the eight magnitudes' patterns (codes 0..7) as prmt's eight-byte tables; the sign bits OR'd in
__device__ __forceinline__ unsigned e2m1x2_permute(unsigned w, int i, unsigned hi_0, unsigned hi_1, unsigned lo_0,
                                                   unsigned lo_1) {
    const unsigned b = (w >> (8 * i)) & 0xffu;
    const unsigned sel = (b & 0x7u) | ((b & 0x70u) << 4);   // the low half's magnitude in nibble 0, the high half's in 2
    unsigned h, l;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(hi_0), "r"(hi_1), "r"(sel << 4));   // bytes 1, 3 (0, 2: code 0's 0x00)
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(l) : "r"(lo_0), "r"(lo_1), "r"(sel));        // bytes 0, 2 (1, 3: code 0's 0x00)
    return h | l | ((b & 0x08u) << 12) | ((b & 0x80u) << 24);
}

/// f16x2 as bf16x2, for values bf16 holds exactly (each half through f32)
__device__ __forceinline__ unsigned f16x2_bf16x2(unsigned f) {
    unsigned h;
    asm("{\n"
        ".reg .f16 l, u;\n"
        ".reg .f32 a, b;\n"
        "mov.b32 {l, u}, %1;\n"
        "cvt.f32.f16 a, l;\n"
        "cvt.f32.f16 b, u;\n"
        "cvt.rn.bf16x2.f32 %0, b, a;\n"
        "}\n"
        : "=r"(h)
        : "r"(f));
    return h;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 through f16x2
__device__ __forceinline__ unsigned e4m3x2_bf16x2_via_f16(unsigned x) {
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f16x2_bf16x2(f);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 through f16x2
__device__ __forceinline__ void e2m1x8_bf16x2_via_f16(unsigned w, unsigned (&h)[4]) {
    unsigned f[4];
    // f16 patterns of 0, 0.5, 1, 1.5, 2, 3, 4, 6: 0x0000 0x3800 0x3c00 0x3e00 0x4000 0x4200 0x4400 0x4600 (low bytes zero)
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = e2m1x2_permute(w, i, 0x3e3c3800u, 0x46444240u, 0u, 0u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = f16x2_bf16x2(f[i]);
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
    return e4m3x2_bf16x2_via_f16(x);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
    // bf16 patterns of 0, 0.5, 1, 1.5, 2, 3, 4, 6: 0x0000 0x3f00 0x3f80 0x3fc0 0x4000 0x4040 0x4080 0x40c0
#pragma unroll
    for (int i = 0; i < 4; ++i)
        h[i] = mul_bf16x2(e2m1x2_permute(w, i, 0x3f3f3f00u, 0x40404040u, 0xc0800000u, 0xc0804000u), sc);
}

}  // namespace basalt::kernels
