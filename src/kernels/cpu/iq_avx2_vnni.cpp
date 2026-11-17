// src/kernels/cpu/iq_avx2_vnni.cpp - iq_avx2_rows.inc built for AVX-VNNI: the same rows, the int16 products summed by the
// one instruction that does the `madd` and the `add`.  Only called behind cpu_avxvnni_ok().

#include "halo/kernels/cpu/iq_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "halo/ggml/ggml_common.h"
#include "halo/log.hpp"
#include "halo/platform/compiler.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#define HALO_IQ_VNNI 1
#define HALO_IQ_BMI2 1
#define HALO_IQ_EXPORT(name) name##_vnni
#include "iq_avx2_rows.inc"
