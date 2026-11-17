// src/models/qwen4exp/kernels/kv_q4.cu - see include/halo/models/qwen4exp/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (with KV-streaming integration and a deterministic group maximum). The block, its quantizer and the rotation are in
// halo/kernels/kv_q4_layout.hpp, shared with the dense path's 4-bit KV formats.
#include "halo/models/qwen4exp/kernels/kv_q4.hpp"
#include "halo/kernels/f16_bits.hpp"
#include "halo/models/qwen4exp/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
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

__device__ __forceinline__ void q4_store(uint8_t* pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// One block = one 32-value group of one KV head of K (blockIdx.z = 0) or V (1); 32 threads. KV streaming: the VRAM
// page only if the block is resident (table >= 0), the host copy always (identity layout) when there is one.
__global__ void kv_append_q4_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                    const float* __restrict__ kcur, const float* __restrict__ vcur,
                                    int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const long long pos = (long long) __ldg(step + kStepPos);
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const bool is_v = blockIdx.z == 1;
    const float x = (is_v ? vcur : kcur)[h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    const uint16_t d = kv_q4_group(x, byte);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
__global__ void kv_append_q4_batch_kernel(uint8_t* __restrict__ k_q4, uint8_t* __restrict__ v_q4,
                                          const int32_t* __restrict__ table, int64_t pos0,
                                          const float* __restrict__ K, const float* __restrict__ V,
                                          int kv_heads, int head_dim, int page_size, int is_v_grid, KvHostPools host,
                                          KvHostPools stage) {
    const long long t = blockIdx.x;
    const long long pos = pos0 + t;
    const int h = blockIdx.y, b = blockIdx.z, th = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    const uint16_t d = kv_q4_group(x, byte);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr) q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr) q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
__global__ void kv_gather_q4_kernel(const uint8_t* __restrict__ k_q4, const uint8_t* __restrict__ v_q4,
                                    const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                    const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                    uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;

    const int t = threadIdx.x;
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        halo::log::error("halo", "%s: head_dim must be 256 (the Hadamard transform's size)", what);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0), 2);
    kv_append_q4_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size,
        host ? *host : KvHostPools{});
    check("kv_append_q4 launch");
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q4_batch_kernel<<<grid, 32, 0, cs>>>(k_q4, v_q4, page_table, pos0, K, V, (int) s.n_head_kv,
                                                       (int) s.head_dim, (int) s.page_size, is_v, h, st);
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q4_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q4, v_q4, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_q4 launch");
}

}  // namespace halo::kernels
