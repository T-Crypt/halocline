// include/basalt/platform/compiler.hpp - the compiler- and CPU-specific spellings the host code needs, behind
// one name each, so no source file asks which compiler it is built with.
//
// x86-64 only: the engine's CPU kernels are AVX2 / AVX-512. Every function is inline and maps to the one instruction
// or builtin each compiler gives for it.
#pragma once

#include <cstdint>

#include <immintrin.h>
#include <cpuid.h>

// Marks a function that uses AVX2, FMA and F16C intrinsics in a translation unit built without those flags. MSVC
// accepts the intrinsics in any function and needs no marking.
#define BASALT_TARGET_AVX2_FMA_F16C __attribute__((target("avx2,fma,f16c")))

namespace basalt::platform {

// MSVC can retain temporary stack copies for vector aggregate assignments even after inlining.
// Field-wise assignment exposes the values to register allocation.
inline constexpr bool kPreferMemberwiseVectorAssignment = false;

/// CPUID leaf `leaf`, subleaf `sub`: r[0..3] = EAX, EBX, ECX, EDX.
inline void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}

/// XCR0, which says which register state the OS saves (read only after CPUID reports OSXSAVE).
inline unsigned long long xgetbv0() {
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
}

/// The index of the lowest set bit of a nonzero mask.
inline int ctz32(unsigned m) {
    return __builtin_ctz(m);
}

/// One iteration of a spin wait: yields the pipeline to the core's SMT sibling. A no-op off x86.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#endif
}

/// The fp32 -> half -> fp32 round trip, round to nearest even. Where the compiler has a half type it does the
/// conversion; MSVC has none, and its F16C intrinsic pair is the same instruction with the same rounding.
inline float f16_round(float x) {
    return (float) (_Float16) x;
}

/// Whether this CPU, and the OS saving its AVX-512 state, can run AVX512_BF16 with AVX512BW.
inline bool cpu_has_avx512_bf16() {
    return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw");
}

}  // namespace basalt::platform
