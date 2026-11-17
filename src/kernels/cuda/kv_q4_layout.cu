// src/kernels/cuda/kv_q4_layout.cu - the rotation the 4-bit KV formats share (see halo/kernels/kv_q4_layout.hpp).
#include "halo/kernels/kv_q4_layout.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include "halo/log.hpp"

namespace halo::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        halo::log::error("halo", "kv_q4: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers. Orthonormal (scale
// 1/sqrt(256) = 1/16), so it is its own inverse. The stages run in index-bit order (1, 2, ... 128), the same order the
// host references use, so a row's sums are that arithmetic and not another one.
__global__ void fwht256_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n_rows, float scale) {
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = __shfl_xor_sync(0xffffffffu, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel<<<dim3((unsigned) num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        src, dst, n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

}  // namespace halo::kernels
