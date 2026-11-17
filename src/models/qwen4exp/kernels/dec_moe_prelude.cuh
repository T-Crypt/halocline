// src/models/qwen4exp/kernels/dec_moe_prelude.cuh - what dec_moe.cu starts from: the shared expert kernels' device code,
// then dec_moe.cuh on top of it.
#pragma once
#define IQ_DEV_ONLY
#include "../../../kernels/cuda/iq_kernels.cu"
#undef IQ_DEV_ONLY

#include "dec_moe.cuh"   // DEC_MOE: the decode MoE tail's device code, on the shared row dots
