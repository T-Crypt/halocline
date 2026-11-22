// src/models/inkling2/vision.cu - Inkling 2 Mini's image encoder (the pack's vision/mmproj, projector inkling2v): a 23-layer
// ViT with a CLS token, learned position embeddings resized to the picture, 2D RoPE, LayerNorm and layer scale, then a
// 2x2 merge and a GELU MLP into the text width. The math follows llama.cpp's mtmd inkling2v graph. The BF16 matrices stay
// in page-locked host memory and go to the card a layer at a time (about 27 MB), so an image needs ~150 MB of VRAM
// beside a full context instead of the encoder's 0.64 GB.
#include "models/inkling2/vision.hpp"
#include "models/inkling2/kernels/ink_ops.hpp"

#include "halo/artifact/gguf_reader.hpp"
#include "halo/log.hpp"
#include "halo/vision/preprocess.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace halo::models::inkling2 {
namespace {

constexpr int kTypeF32 = 0, kTypeBf16 = 30;

unsigned nb(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void layernorm_kernel(const float* x, const float* w, const float* b, float* y, int width, float eps) {
    __shared__ float red[32];
    const float* xr = x + (int64_t) blockIdx.x * width;
    float* yr = y + (int64_t) blockIdx.x * width;
    float s = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) s += xr[i], s2 += xr[i] * xr[i];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o), s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    __shared__ float r2[32];
    if (lane == 0) red[warp] = s, r2[warp] = s2;
    __syncthreads();
    s = 0.f, s2 = 0.f;
    for (int i = 0; i < nw; ++i) s += red[i], s2 += r2[i];
    const float mean = s / width, var = fmaxf(s2 / width - mean * mean, 0.f), inv = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}
__global__ void bias_kernel(float* x, const float* b, int64_t rows, int width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) x[i] += b[i % width];
}
__global__ void scale_add_kernel(float* res, const float* x, const float* s, int64_t rows, int width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) res[i] += x[i] * s[i % width];
}
__global__ void gelu_erf_kernel(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.5f * x[i] * (1.f + erff(x[i] * 0.70710678118654752f));
}
// 2D RoPE (ggml NORM mode, adjacent pairs) on x [T][heads][64]: dims 0..31 at pos_w, 32..63 at pos_h, theta 10000
__global__ void rope2d_kernel(float* x, const int32_t* pw, const int32_t* ph, int T, int heads, float theta) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one pair
    if (i >= (int64_t) T * heads * 32) return;
    const int64_t th = i / 32;
    const int pr = (int) (i % 32), t = (int) (th / heads);
    const int half = pr / 16, j = pr % 16;   // pair j of the half's 32 dims
    const float pos = (float) (half ? ph[t] : pw[t]);
    const float ang = pos * powf(theta, -2.f * j / 32.f);
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float* r = x + th * 64 + half * 32 + 2 * j;
    const float a0 = r[0], a1 = r[1];
    r[0] = a0 * cs - a1 * sn;
    r[1] = a0 * sn + a1 * cs;
}
__global__ void softmax_rows_kernel(float* s, int n) {
    __shared__ float red[32];
    float* r = s + (int64_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, r[i]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int i = 1; i < nw; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float z = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) z += (r[i] = __expf(r[i] - m));
    for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    z = 0.f;
    for (int i = 0; i < nw; ++i) z += red[i];
    const float inv = 1.f / z;
    for (int i = threadIdx.x; i < n; i += blockDim.x) r[i] *= inv;
}
// the 2x2 merge of the patch grid (CLS dropped: x starts at row 1): token (y2, x2) = [(2y2, 2x2), (2y2, 2x2+1),
// (2y2+1, 2x2), (2y2+1, 2x2+1)], each D wide
__global__ void merge2x2_kernel(const float* x, float* out, int wp, int hp, int D) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int w2 = wp / 2, h2 = hp / 2;
    if (i >= (int64_t) w2 * h2 * 4 * D) return;
    const int64_t tok = i / (4 * D), rem = i % (4 * D);
    const int q = (int) (rem / D), d = (int) (rem % D);
    const int y = (int) (tok / w2) * 2 + q / 2, xx = (int) (tok % w2) * 2 + q % 2;
    out[i] = x[(1 + (int64_t) y * wp + xx) * D + d];
}

cublasHandle_t blas() {
    static cublasHandle_t h = nullptr;
    if (!h) {
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH);
    }
    return h;
}

}  // namespace

VisionEncoder::~VisionEncoder() {
    for (void* p : dev_) cudaFree(p);
    if (host_) cudaFreeHost(host_);
}

bool VisionEncoder::load(const GgufFile& g, std::string& err) {
    const auto num = [&](const char* k, double d) { const MetaValue* v = g.get(k); return v && v->is_num() ? v->num() : d; };
    const MetaValue* pt = g.get("clip.vision.projector_type");
    if (!pt || pt->s != "inkling2v") {
        err = "inkling2: the vision projector is not inkling2v";
        return false;
    }
    D_ = (int) num("clip.vision.embedding_length", 1024);
    ff_ = (int) num("clip.vision.feed_forward_length", 4096);
    layers_ = (int) num("clip.vision.block_count", 23);
    heads_ = (int) num("clip.vision.attention.head_count", 16);
    patch_ = (int) num("clip.vision.patch_size", 20);
    merge_ = (int) num("clip.vision.spatial_merge_size", 2);
    eps_ = (float) num("clip.vision.attention.layer_norm_epsilon", 1e-5);
    out_ = (int) num("clip.vision.projection_dim", 2304);
    if (D_ / heads_ != 64 || merge_ != 2) {
        err = "inkling2: a vision encoder shape this engine does not run";
        return false;
    }
    std::map<std::string, const TensorInfo*> t;
    for (const TensorInfo& x : g.tensors()) t[x.name] = &x;
    // BF16 matrices to page-locked host memory, the rest (small) to the card
    size_t host_bytes = 0;
    for (const auto& [n, x] : t)
        if (x->type == (uint32_t) kTypeBf16) host_bytes += tensor_payload_bytes(*x);
    if (cudaHostAlloc(&host_, host_bytes, cudaHostAllocDefault) != cudaSuccess) {
        err = "inkling2: out of page-locked memory for the vision encoder";
        return false;
    }
    size_t at = 0;
    for (const auto& [n, x] : t) {
        const size_t b = tensor_payload_bytes(*x);
        if (x->type == (uint32_t) kTypeBf16) {
            std::memcpy((uint8_t*) host_ + at, g.tensor_data(*x), b);
            hostw_[n] = (const uint16_t*) ((uint8_t*) host_ + at);
            at += b;
        } else if (x->type == (uint32_t) kTypeF32) {
            void* d = nullptr;
            if (cudaMalloc(&d, b) != cudaSuccess) {
                err = "inkling2: out of device memory for the vision encoder's small tensors";
                return false;
            }
            cudaMemcpy(d, g.tensor_data(*x), b, cudaMemcpyHostToDevice);
            dev_.push_back(d);
            devw_[n] = (const float*) d;
        }
    }
    // the position-embedding grid stays on the host too (resized per picture)
    if (const TensorInfo* pe = t.count("v.position_embd.weight") ? t["v.position_embd.weight"] : nullptr) {
        pos_grid_.resize(pe->elements());
        std::memcpy(pos_grid_.data(), g.tensor_data(*pe), pe->elements() * 4);
        grid_ = (int) std::lround(std::sqrt((double) (pe->elements() / D_)));
    }
    for (const char* need : {"v.patch_embd.weight", "v.class_embd", "v.class_position_embd.weight", "v.pre_ln.weight",
                             "v.pre_ln.bias"})
        if (!devw_.count(need)) {
            err = std::string("inkling2: the vision encoder lacks ") + need;
            return false;
        }
    if (!hostw_.count("mm.1.weight") || !hostw_.count("mm.2.weight") || grid_ == 0) {
        err = "inkling2: the vision projector lacks mm.1 / mm.2 or the position grid";
        return false;
    }
    halo::log::info("halo inkling2", "vision encoder: %d layers, width %d, %.0f MiB in page-locked RAM, streamed a layer at a time",
                    layers_, D_, host_bytes / 1048576.0);
    return true;
}

// y [T][N] = x [T][K] . W[N][K]^T for a BF16 W already on the card (k::gemm_bf16)
bool VisionEncoder::encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, int& gx, int& gy,
                           std::string& err, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    vision::Image img;
    if (!vision::decode_image(data, bytes, img, err)) return false;
    // the canvas: a multiple of patch * merge; past the token limit the picture shrinks with its ratio kept
    const int align = patch_ * merge_;
    int w = img.w, h = img.h;
    const int64_t max_px = (int64_t) max_tokens_ * align * align;
    const auto up = [&](int v) { return (v + align - 1) / align * align; };
    if ((int64_t) up(w) * up(h) > max_px) {
        const double sc = std::sqrt((double) max_px / ((double) w * h));
        w = std::max(align, (int) (w * sc) / align * align);
        h = std::max(align, (int) (h * sc) / align * align);
        img = vision::resize_bicubic_to(img, w, h);
    }
    const int W = up(w), H = up(h), wp = W / patch_, hp = H / patch_, np = wp * hp, T = np + 1;
    // patches [np][3 * patch * patch] in (c, ky, kx) order, normalized ((p/255 - 0.5)/0.5), the pad at the normalized -1
    const int pk = 3 * patch_ * patch_;
    std::vector<float> patches((size_t) np * pk);
    const float pad = (-1.f - 127.5f) / 127.5f;
    for (int py = 0; py < hp; ++py)
        for (int px = 0; px < wp; ++px) {
            float* o = patches.data() + ((size_t) py * wp + px) * pk;
            for (int c = 0; c < 3; ++c)
                for (int ky = 0; ky < patch_; ++ky)
                    for (int kx = 0; kx < patch_; ++kx) {
                        const int x = px * patch_ + kx, y = py * patch_ + ky;
                        o[(c * patch_ + ky) * patch_ + kx] =
                            (x < img.w && y < img.h) ? ((float) img.rgb[((size_t) y * img.w + x) * 3 + c] - 127.5f) / 127.5f : pad;
                    }
        }
    // position embeddings: the grid resized bilinearly (half-pixel) to wp x hp; the CLS row's is its own
    std::vector<float> pos((size_t) np * D_);
    for (int y = 0; y < hp; ++y)
        for (int x = 0; x < wp; ++x) {
            const float sx = std::clamp((x + 0.5f) * grid_ / wp - 0.5f, 0.f, (float) grid_ - 1);
            const float sy = std::clamp((y + 0.5f) * grid_ / hp - 0.5f, 0.f, (float) grid_ - 1);
            const int x0 = (int) sx, y0 = (int) sy, x1 = std::min(x0 + 1, grid_ - 1), y1 = std::min(y0 + 1, grid_ - 1);
            const float fx = sx - x0, fy = sy - y0;
            const float* g00 = pos_grid_.data() + ((size_t) y0 * grid_ + x0) * D_;
            const float* g01 = pos_grid_.data() + ((size_t) y0 * grid_ + x1) * D_;
            const float* g10 = pos_grid_.data() + ((size_t) y1 * grid_ + x0) * D_;
            const float* g11 = pos_grid_.data() + ((size_t) y1 * grid_ + x1) * D_;
            float* o = pos.data() + ((size_t) y * wp + x) * D_;
            for (int d = 0; d < D_; ++d)
                o[d] = (1 - fy) * ((1 - fx) * g00[d] + fx * g01[d]) + fy * ((1 - fx) * g10[d] + fx * g11[d]);
        }
    std::vector<int32_t> pw((size_t) T), ph((size_t) T);
    pw[0] = ph[0] = 0;
    for (int y = 0; y < hp; ++y)
        for (int x = 0; x < wp; ++x) pw[(size_t) 1 + y * wp + x] = x + 1, ph[(size_t) 1 + y * wp + x] = y + 1;
    // device scratch for this picture
    const int D = D_, F = ff_, NT = (wp / 2) * (hp / 2), heads = heads_;
    const int64_t att_chunk = std::max<int64_t>(1, std::min<int64_t>(heads, (64ll << 20) / ((int64_t) T * T * 4)));   // scores <= ~64 MB
    std::vector<void*> tmp;
    const auto take = [&](size_t b) -> float* {
        void* p = nullptr;
        if (cudaMalloc(&p, b) != cudaSuccess) return nullptr;
        tmp.push_back(p);
        return (float*) p;
    };
    struct Free {
        std::vector<void*>& v;
        ~Free() { for (void* p : v) cudaFree(p); }
    } fr{tmp};
    const size_t layer_w = (size_t) (4 * D * D + 2 * D * F) * 2;
    float *dp = take((size_t) np * pk * 4), *x = take((size_t) T * D * 4), *hbuf = take((size_t) T * D * 4),
          *q = take((size_t) T * D * 4), *k = take((size_t) T * D * 4), *v = take((size_t) T * D * 4), *o = take((size_t) T * D * 4),
          *u = take((size_t) T * F * 4), *sc = take((size_t) att_chunk * T * T * 4), *mrg = take((size_t) NT * 4 * D * 4),
          *mm1 = take((size_t) NT * 4 * D * 4), *mm2 = take((size_t) NT * out_ * 4);
    uint16_t* wl = (uint16_t*) take(std::max(layer_w, (size_t) (16 * D * D + 4 * D * out_) * 2));
    int32_t *dpw = (int32_t*) take((size_t) T * 4), *dph = (int32_t*) take((size_t) T * 4);
    if (!dp || !x || !hbuf || !q || !k || !v || !o || !u || !sc || !mrg || !mm1 || !mm2 || !wl || !dpw || !dph) {
        err = "inkling2: no device memory for this picture's encoder pass";
        return false;
    }
    cudaMemcpyAsync(dp, patches.data(), patches.size() * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dpw, pw.data(), (size_t) T * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dph, ph.data(), (size_t) T * 4, cudaMemcpyHostToDevice, s);
    // patch embedding (F32) into rows 1..; the CLS row = class_embd + class_position_embd; then + position embeddings
    const float one = 1.f, zero = 0.f;
    cublasSetStream(blas(), s);
    cublasSgemm(blas(), CUBLAS_OP_T, CUBLAS_OP_N, D, np, pk, &one, devw_["v.patch_embd.weight"], pk, dp, pk, &zero, x + D, D);
    cudaMemcpyAsync(hbuf, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice, s);
    k::add2(x + D, hbuf, x + D, (int64_t) np * D, s);
    k::add2(devw_["v.class_embd"], devw_["v.class_position_embd.weight"], x, D, s);
    layernorm_kernel<<<(unsigned) T, 256, 0, s>>>(x, devw_["v.pre_ln.weight"], devw_["v.pre_ln.bias"], x, D, eps_);
    const float kq = 1.f / std::sqrt((float) (D / heads));
    for (int l = 0; l < layers_; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        // this layer's matrices to the card
        uint16_t* wq = wl;
        uint16_t* wk = wq + (size_t) D * D;
        uint16_t* wv = wk + (size_t) D * D;
        uint16_t* wo = wv + (size_t) D * D;
        uint16_t* wu = wo + (size_t) D * D;
        uint16_t* wd = wu + (size_t) F * D;
        const std::pair<uint16_t*, const char*> mats[] = {{wq, "attn_q.weight"}, {wk, "attn_k.weight"}, {wv, "attn_v.weight"},
                                                          {wo, "attn_out.weight"}, {wu, "ffn_up.weight"}, {wd, "ffn_down.weight"}};
        for (const auto& [dst, n] : mats) {
            auto it = hostw_.find(p + n);
            if (it == hostw_.end()) {
                err = "inkling2: the vision encoder lacks " + p + n;
                return false;
            }
            const size_t b = (std::string(n) == "ffn_up.weight" || std::string(n) == "ffn_down.weight") ? (size_t) F * D * 2 : (size_t) D * D * 2;
            cudaMemcpyAsync(dst, it->second, b, cudaMemcpyHostToDevice, s);
        }
        layernorm_kernel<<<(unsigned) T, 256, 0, s>>>(x, devw_[p + "ln1.weight"], devw_[p + "ln1.bias"], hbuf, D, eps_);
        k::gemm_bf16(wq, hbuf, q, T, D, D, s);
        k::gemm_bf16(wk, hbuf, k, T, D, D, s);
        k::gemm_bf16(wv, hbuf, v, T, D, D, s);
        bias_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(q, devw_[p + "attn_q.bias"], T, D);
        bias_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(k, devw_[p + "attn_k.bias"], T, D);
        bias_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(v, devw_[p + "attn_v.bias"], T, D);
        rope2d_kernel<<<nb((int64_t) T * heads * 32), 256, 0, s>>>(q, dpw, dph, T, heads, 10000.f);
        rope2d_kernel<<<nb((int64_t) T * heads * 32), 256, 0, s>>>(k, dpw, dph, T, heads, 10000.f);
        cublasSetStream(blas(), s);
        for (int h0 = 0; h0 < heads; h0 += (int) att_chunk) {
            const int nh = std::min<int>((int) att_chunk, heads - h0);
            // S_h [T][T] = Q_h K_h^T * scale; softmax; O_h = P V_h   (head h at column 64 h of the [T][D] rows)
            cublasSgemmStridedBatched(blas(), CUBLAS_OP_T, CUBLAS_OP_N, T, T, 64, &kq, k + h0 * 64, D, 64, q + h0 * 64, D, 64, &zero,
                                      sc, T, (long long) T * T, nh);
            softmax_rows_kernel<<<(unsigned) ((int64_t) nh * T), 256, 0, s>>>(sc, T);
            cublasSgemmStridedBatched(blas(), CUBLAS_OP_N, CUBLAS_OP_N, 64, T, T, &one, v + h0 * 64, D, 64, sc, T, (long long) T * T,
                                      &zero, o + h0 * 64, D, 64, nh);
        }
        k::gemm_bf16(wo, o, hbuf, T, D, D, s);
        bias_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(hbuf, devw_[p + "attn_out.bias"], T, D);
        scale_add_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(x, hbuf, devw_[p + "ls1.weight"], T, D);
        layernorm_kernel<<<(unsigned) T, 256, 0, s>>>(x, devw_[p + "ln2.weight"], devw_[p + "ln2.bias"], hbuf, D, eps_);
        k::gemm_bf16(wu, hbuf, u, T, F, D, s);
        bias_kernel<<<nb((int64_t) T * F), 256, 0, s>>>(u, devw_[p + "ffn_up.bias"], T, F);
        gelu_erf_kernel<<<nb((int64_t) T * F), 256, 0, s>>>(u, (int64_t) T * F);
        k::gemm_bf16(wd, u, hbuf, T, D, F, s);
        bias_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(hbuf, devw_[p + "ffn_down.bias"], T, D);
        scale_add_kernel<<<nb((int64_t) T * D), 256, 0, s>>>(x, hbuf, devw_[p + "ls2.weight"], T, D);
    }
    // merge, project
    merge2x2_kernel<<<nb((int64_t) NT * 4 * D), 256, 0, s>>>(x, mrg, wp, hp, D);
    uint16_t* w1 = wl;
    uint16_t* w2 = w1 + (size_t) 16 * D * D;
    cudaMemcpyAsync(w1, hostw_["mm.1.weight"], (size_t) 16 * D * D * 2, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(w2, hostw_["mm.2.weight"], (size_t) 4 * D * out_ * 2, cudaMemcpyHostToDevice, s);
    k::gemm_bf16(w1, mrg, mm1, NT, 4 * D, 4 * D, s);
    if (devw_.count("mm.1.bias")) bias_kernel<<<nb((int64_t) NT * 4 * D), 256, 0, s>>>(mm1, devw_["mm.1.bias"], NT, 4 * D);
    gelu_erf_kernel<<<nb((int64_t) NT * 4 * D), 256, 0, s>>>(mm1, (int64_t) NT * 4 * D);
    k::gemm_bf16(w2, mm1, mm2, NT, out_, 4 * D, s);
    if (devw_.count("mm.2.bias")) bias_kernel<<<nb((int64_t) NT * out_), 256, 0, s>>>(mm2, devw_["mm.2.bias"], NT, out_);
    rows.resize((size_t) NT * out_);
    cudaMemcpyAsync(rows.data(), mm2, rows.size() * 4, cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) {
        err = std::string("inkling2: the vision encoder failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    n_rows = NT;
    gx = wp / 2;
    gy = hp / 2;
    return true;
}

}  // namespace halo::models::inkling2
