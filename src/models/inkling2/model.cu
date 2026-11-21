// src/models/inkling2/model.cu - Inkling 2 Mini on one card: the weights as the GGUF stores them, the 48-visit forward,
// and the session the server drives. First light (halo-inkling): correctness first, every visit through the same path
// for decode windows and prompt chunks; the products run on the engine's MMQ (Q8_0 projections, NVFP4 and Q8_0
// experts grouped by expert) and native mmvq (Q8_0 decode rows), the rest on the model's own kernels (kernels/ink_ops).
#include "models/inkling2/model.hpp"
#include "models/inkling2/kernels/ink_ops.hpp"

#include "halo/artifact/gguf_reader.hpp"
#include "halo/core/load_progress.hpp"
#include "halo/core/model_file.hpp"
#include "halo/kernels/iq_kernels.hpp"
#include "halo/kernels/logprobs.hpp"
#include "halo/kernels/native_mmvq.hpp"
#include "halo/kernels/sampler.hpp"
#include "halo/log.hpp"
#include "halo/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace halo::models::inkling2 {
namespace {

namespace mmq = prefill::mmq;
using model::Result;
using model::Status;

constexpr int kChunk = 512;    // prompt rows per forward (the SWA ring needs chunk + window <= ring)
constexpr int kWindow = 16;    // decode window rows
constexpr int kRing = 1536;    // cells of a sliding-window visit's cache (window 1024 + a 512-row chunk)
constexpr int kMmvqRows = 4;   // Q8_0 products of at most this many rows on native mmvq
constexpr int kDecodeExperts = 4;   // MoE of at most this many rows on the decode expert kernels

// HALO_INK_PROFILE=1: per-stage GPU time of the forward (CUDA events), printed every 32 forwards
struct Prof {
    bool on = std::getenv("HALO_INK_PROFILE") != nullptr;
    static constexpr int kStages = 8;
    const char* names[kStages] = {"embed", "mix+ctl", "proj", "attn", "attn-misc", "moe-route", "moe-gemm", "head"};
    double ms[kStages] = {};
    int forwards = 0;
    std::vector<std::pair<int, cudaEvent_t>> marks;
    std::vector<cudaEvent_t> pool;
    size_t used = 0;
    bool capturing = false;
    void mark(int stage, cudaStream_t s) {
        if (!on || capturing) return;
        if (used == pool.size()) {
            cudaEvent_t e;
            cudaEventCreate(&e);
            pool.push_back(e);
        }
        cudaEventRecord(pool[used], s);
        marks.push_back({stage, pool[used++]});
    }
    void end(cudaStream_t s) {
        if (!on || capturing) return;
        mark(-1, s);
        cudaEventSynchronize(marks.back().second);
        for (size_t i = 0; i + 1 < marks.size(); ++i) {
            float t = 0;
            cudaEventElapsedTime(&t, marks[i].second, marks[i + 1].second);
            if (marks[i].first >= 0) ms[marks[i].first] += t;
        }
        marks.clear();
        used = 0;
        if (++forwards % 32 == 0) {
            double tot = 0;
            for (double v : ms) tot += v;
            std::fprintf(stderr, "ink profile over %d forwards (ms/forward):", forwards);
            for (int i = 0; i < kStages; ++i) std::fprintf(stderr, " %s %.3f", names[i], ms[i] / forwards);
            std::fprintf(stderr, " | total %.3f\n", tot / forwards);
        }
    }
};

// ---- small kernels of this file -------------------------------------------------------------------------------------

// y[t][o] = dot(W[o], x[t]) for an FP32 matrix W [out][in]: one warp per output value.
__global__ void f32_matvec_kernel(const float* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int T,
                                  int in, int out) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (warp >= T * out) return;
    const int t = warp / out, o = warp % out;
    const float* w = W + (size_t) o * in;
    const float* xr = x + (size_t) t * in;
    float s = 0.f;
    for (int i = lane; i < in; i += 32) s += w[i] * xr[i];
    for (int off = 16; off; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if (lane == 0) y[(size_t) t * out + o] = s;
}
void f32_matvec(const float* W, const float* x, float* y, int T, int in, int out, cudaStream_t s) {
    const int warps = T * out, threads = 256;
    f32_matvec_kernel<<<(warps * 32 + threads - 1) / threads, threads, 0, s>>>(W, x, y, T, in, out);
}
// rows of `width` copied out of rows of `ld` (the padded head into the unpadded logits)
__global__ void copy_rows_kernel(const float* src, int64_t ld, float* dst, int64_t width, int rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * rows) return;
    dst[i] = src[(i / width) * ld + i % width];
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- weights ----------------------------------------------------------------------------------------------------------

struct Lin {
    int type = -1;
    const void* w = nullptr;
    int64_t rows = 0, cols = 0;   // out, in
};

struct Experts {
    int type = -1;
    const void* gu = nullptr;     // [n_expert][2 n_ff][n_embd]
    const void* down = nullptr;   // [n_expert][n_embd][n_ff]
    const float* gu_scale = nullptr;
    const float* d_scale = nullptr;
    size_t gu_bytes = 0, d_bytes = 0;   // one expert's
};

struct Layer {
    bool swa = true;
    const float *attn_norm = nullptr, *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;
    Lin q, k, v, o, gate, rel;
    const float* sink = nullptr;           // F32 [n_head][n_embd] (from BF16)
    const float *k_cp = nullptr, *v_cp = nullptr;   // F32 [conv * n_kv][n_embd] (from BF16), full layers
    const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    const float *k_conv = nullptr, *v_conv = nullptr, *post_conv = nullptr, *ffn_post_conv = nullptr;
    const float* attn_res_query = nullptr;   // [1 or 3][n_embd], null for layer 0
    int n_attn_q = 0;
    const float* ffn_res_query = nullptr;
    const float* ffn_norm = nullptr;
    const float* router = nullptr;           // F32 [n_expert][n_embd]
    const float* probs_b = nullptr;          // [n_expert]
    Experts ex;
    const float *loop_down[4] = {}, *loop_gain[4] = {}, *loop_forget[4] = {};
};

struct Weights {
    const void* embd = nullptr;   // mapped host, Q8_0 rows
    int embd_type = -1;
    size_t embd_row_bytes = 0;
    const float* embd_norm = nullptr;
    Lin head;
    const float *out_norm = nullptr, *out_res_query = nullptr;
    const float *loop_res_query = nullptr, *loop_kv_res_query = nullptr;
    std::vector<Layer> layers;   // stored layers
};

// ---- the model ----------------------------------------------------------------------------------------------------

class Model;

// Per-session state: K/V and convolution state per visit, the scratch of one forward.
struct State {
    cudaStream_t s = nullptr;
    std::vector<uint16_t*> kc, vc;   // per visit
    std::vector<int64_t> ring;       // per visit: cells when a ring, 0 linear
    float* conv_state = nullptr;     // per visit: [k (n_kv hd) | v (n_kv hd) | attn out (D) | ffn out (D)] x (conv - 1)
    int64_t conv_floats = 0;         // per visit
    // forward scratch, sized for kChunk rows
    float *cur, *inp_sa, *prelude_cur, *bank_last, *in3[3], *x, *q, *k, *v, *r, *g, *sink, *kdyn, *vdyn, *att, *y, *msg[4];
    float* blocks[10];
    float* tau;
    float *rlogits, *rweight, *gu, *h, *dn, *moe_out;
    int32_t *sel, *bounds, *pairs, *row_of, *src_tok, *expert_of, *iota, *tok, *picks;
    void *xq, *xq2;
    float *logits_pad, *logits;
    int64_t* d_pos = nullptr;            // the decode graph's position
    int64_t* h_pos = nullptr;            // pinned
    int32_t* h_tok = nullptr;            // pinned
    cudaGraphExec_t graph = nullptr;     // the one-row decode forward
    int graph_state = 0;                 // 0 not yet (warm-up next), 1 warm, 2 captured, -1 off
    void* attn_ws = nullptr;
    size_t attn_ws_bytes = 0;
    std::vector<void*> owned;
    ~State() {
        if (graph) cudaGraphExecDestroy(graph);
        if (h_pos) cudaFreeHost(h_pos);
        if (h_tok) cudaFreeHost(h_tok);
        for (void* p : owned) cudaFree(p);
        if (s) cudaStreamDestroy(s);
    }
};

class Model final : public model::LoadedModel {
public:
    ~Model() override {
        if (block_) cudaFree(block_);
        if (embd_host_) cudaFreeHost(embd_host_);
    }
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return max_ctx_; }
    Result make_session(std::unique_ptr<model::ModelSession>& out) override;

    bool load(const model::LoadRequest& req, std::string& err);

    Hparams h;
    Weights w;
    int64_t max_ctx_ = 0;
    std::unique_ptr<mmq::Context> mmq_ctx;

private:
    model::ModelDesc desc_;
    std::map<std::string, TensorInfo> tensors_;
    char* block_ = nullptr;
    uint64_t block_size_ = 0, used_ = 0;
    void* embd_host_ = nullptr;
    const GgufFile* gg_ = nullptr;

    const TensorInfo* find(const std::string& name, std::string& err) {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            err = "inkling2: the file has no " + name;
            return nullptr;
        }
        return &it->second;
    }
    void* take(uint64_t bytes) {
        const uint64_t need = (bytes + 255) & ~(uint64_t) 255;
        if (used_ + need > block_size_) return nullptr;
        void* p = block_ + used_;
        used_ += need;
        return p;
    }
    void* upload_bytes(const void* src, uint64_t bytes, const std::string& name, std::string& err) {
        void* d = take(bytes);
        if (!d) {
            err = "inkling2: " + name + " does not fit the weight block";
            return nullptr;
        }
        if (cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "inkling2: uploading " + name + " failed";
            return nullptr;
        }
        return d;
    }
    const void* raw(const std::string& name, std::string& err, int* type = nullptr) {
        const TensorInfo* t = find(name, err);
        if (!t) return nullptr;
        if (type) *type = (int) t->type;
        return upload_bytes(gg_->tensor_data(*t), tensor_payload_bytes(*t), name, err);
    }
    // an F32 tensor, or a BF16 one widened to F32, of `elems` values
    const float* f32(const std::string& name, int64_t elems, std::string& err) {
        const TensorInfo* t = find(name, err);
        if (!t) return nullptr;
        if ((int64_t) t->elements() != elems) {
            err = "inkling2: " + name + " has " + std::to_string((long long) t->elements()) + " values, not " + std::to_string((long long) elems);
            return nullptr;
        }
        if (t->type == (uint32_t) kTypeF32) return (const float*) upload_bytes(gg_->tensor_data(*t), elems * 4, name, err);
        if (t->type == (uint32_t) kTypeBf16) {
            std::vector<float> v((size_t) elems);
            const uint16_t* b = (const uint16_t*) gg_->tensor_data(*t);
            for (int64_t i = 0; i < elems; ++i) v[(size_t) i] = bf16_to_f32(b[i]);
            return (const float*) upload_bytes(v.data(), elems * 4, name, err);
        }
        err = "inkling2: " + name + " is " + t->type_name() + ", not F32 or BF16";
        return nullptr;
    }
    bool lin(const std::string& name, int64_t rows, int64_t cols, Lin& l, std::string& err) {
        const TensorInfo* t = find(name + ".weight", err);
        if (!t) return false;
        if ((int64_t) t->elements() != rows * cols || t->shape.size() != 2 || (int64_t) t->shape[0] != cols) {
            err = "inkling2: " + name + " is not [" + std::to_string((long long) rows) + "][" + std::to_string((long long) cols) + "]";
            return false;
        }
        if (t->type != (uint32_t) kTypeQ8_0) {
            err = "inkling2: " + name + " is " + t->type_name() + "; first light runs Q8_0 projections only";
            return false;
        }
        l.type = (int) t->type;
        l.rows = rows;
        l.cols = cols;
        l.w = raw(name + ".weight", err);
        return l.w != nullptr;
    }
    bool layer(int il, Layer& L, std::string& err);
};

bool Model::layer(int il, Layer& L, std::string& err) {
    const int D = h.n_embd, AQ = h.n_head * h.head_dim, KV = h.n_kv * h.head_dim, C = h.conv;
    const std::string p = "blk." + std::to_string(il) + ".";
    // a stored layer is SWA as its first visit is (every visit of a stored layer agrees)
    int visit = il < h.core_start ? il : il < h.core_end ? il : il + (h.core_end - h.core_start) * (h.n_pass - 1);
    L.swa = h.is_swa[(size_t) visit] != 0;
    if (!lin(p + "attn_q", AQ, D, L.q, err) || !lin(p + "attn_k", KV, D, L.k, err) || !lin(p + "attn_v", KV, D, L.v, err) ||
        !lin(p + "attn_output", D, AQ, L.o, err) || !lin(p + "attn_gate", AQ, D, L.gate, err) ||
        !lin(p + "attn_rel", (int64_t) h.rel_len * h.n_head, D, L.rel, err))
        return false;
    L.sink = f32(p + "attn_sink_proj.weight", (int64_t) h.n_head * D, err);
    L.q_clip = f32(p + "attn_q_clip.weight", (int64_t) h.n_head * h.head_dim, err);
    L.rel_bias = f32(p + "attn_rel_bias.weight", (int64_t) h.rel_len * (L.swa ? h.swa : h.rel_extent), err);
    L.rel_clip = f32(p + "attn_rel_clip.weight", (int64_t) h.rel_len * h.n_head, err);
    L.k_conv = f32(p + "attn_k_conv.weight", (int64_t) C * KV, err);
    L.v_conv = f32(p + "attn_v_conv.weight", (int64_t) C * KV, err);
    L.post_conv = f32(p + "attn_post_conv.weight", (int64_t) C * D, err);
    L.ffn_post_conv = f32(p + "ffn_post_conv.weight", (int64_t) C * D, err);
    L.ffn_res_query = f32(p + "ffn_res_query.weight", D, err);
    L.ffn_norm = f32(p + "ffn_norm.weight", D, err);
    L.router = f32(p + "ffn_gate_inp.weight", (int64_t) h.n_expert * D, err);
    L.probs_b = f32(p + "exp_probs_b.bias", h.n_expert, err);
    if (!L.sink || !L.q_clip || !L.rel_bias || !L.rel_clip || !L.k_conv || !L.v_conv || !L.post_conv || !L.ffn_post_conv ||
        !L.ffn_res_query || !L.ffn_norm || !L.router || !L.probs_b)
        return false;
    if (L.swa) {
        if (!(L.attn_norm = f32(p + "attn_norm.weight", D, err))) return false;
    } else {
        L.q_inp_norm = f32(p + "attn_q_inp_norm.weight", D, err);
        L.k_inp_norm = f32(p + "attn_k_inp_norm.weight", D, err);
        L.v_inp_norm = f32(p + "attn_v_inp_norm.weight", D, err);
        L.k_cp = f32(p + "attn_k_conv_proj.weight", (int64_t) C * h.n_kv * D, err);
        L.v_cp = f32(p + "attn_v_conv_proj.weight", (int64_t) C * h.n_kv * D, err);
        if (!L.q_inp_norm || !L.k_inp_norm || !L.v_inp_norm || !L.k_cp || !L.v_cp) return false;
    }
    if (il > 0) {
        L.n_attn_q = L.swa ? 1 : 3;
        if (!(L.attn_res_query = f32(p + "attn_res_query.weight", (int64_t) L.n_attn_q * D, err))) return false;
    }
    // experts
    const TensorInfo* tgu = find(p + "ffn_gate_up_exps.weight", err);
    const TensorInfo* td = tgu ? find(p + "ffn_down_exps.weight", err) : nullptr;
    if (!td) return false;
    if (tgu->type != td->type || !mmq::supported((int) tgu->type) || !mmq::fits((int) tgu->type, 2 * h.n_ff) ||
        !mmq::fits((int) td->type, D)) {
        err = "inkling2: " + p + "experts are " + tgu->type_name() + "/" + td->type_name() + ", which MMQ does not run here";
        return false;
    }
    L.ex.type = (int) tgu->type;
    L.ex.gu_bytes = mmq::matrix_bytes(L.ex.type, 2 * h.n_ff, D);
    L.ex.d_bytes = mmq::matrix_bytes(L.ex.type, D, h.n_ff);
    if (L.ex.gu_bytes * h.n_expert != tensor_payload_bytes(*tgu) || L.ex.d_bytes * h.n_expert != tensor_payload_bytes(*td)) {
        err = "inkling2: " + p + "experts are not whole matrices of their type";
        return false;
    }
    L.ex.gu = raw(p + "ffn_gate_up_exps.weight", err);
    L.ex.down = L.ex.gu ? raw(p + "ffn_down_exps.weight", err) : nullptr;
    if (!L.ex.down) return false;
    if (tensors_.count(p + "ffn_gate_up_exps.scale")) {
        L.ex.gu_scale = f32(p + "ffn_gate_up_exps.scale", h.n_expert, err);
        L.ex.d_scale = f32(p + "ffn_down_exps.scale", h.n_expert, err);
        if (!L.ex.gu_scale || !L.ex.d_scale) return false;
    }
    if (il >= h.core_start && il < h.core_end && h.n_pass > 1) {
        static const char* mod[4] = {"attn", "k", "v", "mlp"};
        for (int m = 0; m < 4; ++m) {
            const std::string b = p + "loop_" + mod[m] + "_";
            L.loop_down[m] = f32(b + "down.weight", (int64_t) h.gate_rank * 2 * D, err);
            L.loop_gain[m] = f32(b + "gain.weight", (int64_t) D * h.gate_rank, err);
            L.loop_forget[m] = f32(b + "forget.weight", (int64_t) D * h.gate_rank, err);
            if (!L.loop_down[m] || !L.loop_gain[m] || !L.loop_forget[m]) return false;
        }
    }
    return true;
}

bool Model::load(const model::LoadRequest& req, std::string& err) {
    const std::string& path = req.paths.front();
    if (path.size() > 5 && path.compare(path.size() - 5, 5, ".halo") == 0) {   // a pack: its gguf section as a GGUF
        core::ModelFile mf;
        if (!mf.open(path, err) || !mf.load_meta(err) || !mf.register_gguf_view(err)) return false;
    }
    GgufFile gg(path);
    gg_ = &gg;
    if (!read_hparams(gg, h, err)) return false;
    max_ctx_ = req.max_context > 0 ? req.max_context : 131072;
    if (h.context_length > 0) max_ctx_ = std::min<int64_t>(max_ctx_, h.context_length);
    desc_.arch = "inkling2";
    desc_.n_layers = h.n_visit;
    desc_.n_embd = h.n_embd;
    desc_.n_vocab = h.n_vocab;
    desc_.n_expert = h.n_expert;
    desc_.n_expert_used = h.n_used;
    desc_.n_ff_exp = h.n_ff;
    desc_.context_length = h.context_length;
    uint64_t bytes = 0;
    for (const TensorInfo& t : gg.tensors()) {
        tensors_[t.name] = t;
        if (t.name == "token_embd.weight") continue;
        // F32 widening of BF16 doubles them
        bytes += ((t.type == (uint32_t) kTypeBf16 ? t.elements() * 4 : tensor_payload_bytes(t)) + 255) & ~(uint64_t) 255;
    }
    if (cudaMalloc((void**) &block_, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "inkling2: out of device memory for " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB of weights";
        return false;
    }
    block_size_ = bytes;
    const int D = h.n_embd;
    // the token embedding in page-locked host memory as stored; the kernel reads the rows it needs
    const TensorInfo* te = find("token_embd.weight", err);
    if (!te) return false;
    w.embd_type = (int) te->type;
    if (!kernels::embed_type_supported(w.embd_type)) {
        err = std::string("inkling2: token_embd.weight is ") + te->type_name() + ", which the embedding kernel does not read";
        return false;
    }
    w.embd_row_bytes = kernels::iq_row_bytes(w.embd_type, D);
    if (cudaHostAlloc(&embd_host_, tensor_payload_bytes(*te), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        err = "inkling2: out of page-locked memory for the token embedding";
        return false;
    }
    std::memcpy(embd_host_, gg.tensor_data(*te), tensor_payload_bytes(*te));
    void* dev_embd = nullptr;
    cudaHostGetDevicePointer(&dev_embd, embd_host_, 0);
    w.embd = dev_embd;
    w.embd_norm = f32("token_embd_norm.weight", D, err);
    w.out_norm = f32("output_norm.weight", D, err);
    w.out_res_query = f32("output_res_query.weight", D, err);
    if (!w.embd_norm || !w.out_norm || !w.out_res_query || !lin("output", h.n_vocab_padded, D, w.head, err)) return false;
    if (h.n_pass > 1) {
        const int64_t nq = 2 * (int64_t) (h.core_end - h.core_start) * D;
        w.loop_res_query = f32("loop_res_query.weight", nq, err);
        w.loop_kv_res_query = f32("loop_kv_res_query.weight", nq, err);
        if (!w.loop_res_query || !w.loop_kv_res_query) return false;
    }
    w.layers.resize((size_t) h.n_phys);
    for (int il = 0; il < h.n_phys; ++il)
        if (!layer(il, w.layers[(size_t) il], err)) return false;
    mmq_ctx = std::make_unique<mmq::Context>();
    gg_ = nullptr;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    halo::log::info("halo inkling2", "%zu tensors, %.2f GiB on the GPU, %.2f GiB free; %d visits of %d stored layers, context %lld",
                    tensors_.size(), (double) used_ / 1073741824.0, (double) free_b / 1073741824.0, h.n_visit, h.n_phys,
                    (long long) max_ctx_);
    return true;
}

// ---- the session ------------------------------------------------------------------------------------------------

class Session final : public model::ModelSession {
public:
    explicit Session(Model& m) : m_(m), h_(m.h) {}

    bool init(std::string& err);

    Result decode_window(const model::DecodeRequest& r) override;
    Result prefill_chunk(const model::PrefillRequest& r) override;
    int max_window() const override { return 1; }
    Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_T_) return {Status::Error, "inkling2: no such row"};
        if (cudaMemcpy(out, st_.logits + (size_t) row * h_.n_vocab, (size_t) h_.n_vocab * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            return {Status::Error, "inkling2: the logits copy failed"};
        return {};
    }
    std::string info_words() const override { return "arch=inkling2 visits=" + std::to_string(h_.n_visit); }

    // The running state: every visit's convolution state and the sliding-window rings (overwritten as positions pass,
    // so they cannot be rebuilt from cells). The full-attention K/V is positional and stays valid below the position.
    size_t running_bytes() const override {
        size_t n = sizeof(int64_t) + (size_t) st_.conv_floats * h_.n_visit * 4;
        for (int v = 0; v < h_.n_visit; ++v)
            if (st_.ring[(size_t) v]) n += 2 * (size_t) st_.ring[(size_t) v] * h_.n_kv * h_.head_dim * 2;
        return n;
    }
    Result running_state(int64_t pos, bool save, core::RunningState& rs) override {
        rs.head.resize(sizeof(int64_t));
        if (save) std::memcpy(rs.head.data(), &pos, sizeof pos);
        rs.regions.clear();
        rs.regions.push_back({st_.conv_state, (size_t) st_.conv_floats * h_.n_visit * 4});
        for (int v = 0; v < h_.n_visit; ++v)
            if (const int64_t cells = st_.ring[(size_t) v]) {
                const size_t b = (size_t) cells * h_.n_kv * h_.head_dim * 2;
                rs.regions.push_back({st_.kc[(size_t) v], b});
                rs.regions.push_back({st_.vc[(size_t) v], b});
            }
        rs.stream = st_.s;
        if (save && cudaStreamSynchronize(st_.s) != cudaSuccess) return {Status::Error, "inkling2: the stream failed before a checkpoint"};
        return {};
    }
    bool parks_cells() const override { return false; }

    Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        if (!device_floats(d_bias_, bias)) return {Status::Error, "inkling2: the logit bias upload failed"};
        return {};
    }
    Result resample_row(int row, const float* bias, int32_t& token) override {
        if (row < 0 || row >= last_T_) return {Status::Error, "inkling2: no such row"};
        if (!device_floats(d_mask_, bias)) return {Status::Error, "inkling2: the mask upload failed"};
        kernels::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        sp.n_drawn = 0;
        kernels::sample_tokens(st_.logits + (size_t) row * h_.n_vocab, 1, (int) h_.n_vocab, nullptr, 0, sp, st_.picks, st_.s);
        if (cudaMemcpyAsync(h_picks_, st_.picks, 4, cudaMemcpyDeviceToHost, st_.s) != cudaSuccess ||
            cudaStreamSynchronize(st_.s) != cudaSuccess)
            return {Status::Error, "inkling2: the masked draw failed"};
        token = h_picks_[0];
        return {};
    }
    Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (n <= 0 || row0 < 0 || row0 + n > last_T_) return {Status::Error, "inkling2: row_logprobs rows out of range"};
        if (!lp_buf_) {
            const size_t f = kernels::row_logprobs_scratch_floats(kWindow, (int) h_.n_vocab) + (size_t) kWindow * (kernels::kLogprobStride + 1);
            if (cudaMalloc((void**) &lp_buf_, f * sizeof(float)) != cudaSuccess) return {Status::Error, "inkling2: out of memory for logprobs"};
            st_.owned.push_back(lp_buf_);
        }
        int32_t* d_picks = (int32_t*) (lp_buf_ + kernels::row_logprobs_scratch_floats(kWindow, (int) h_.n_vocab));
        float* d_out = (float*) (d_picks + kWindow);
        cudaMemcpyAsync(d_picks, picks, (size_t) n * 4, cudaMemcpyHostToDevice, st_.s);
        kernels::row_logprobs(st_.logits + (size_t) row0 * h_.n_vocab, n, (int) h_.n_vocab, d_picks, top, d_out, lp_buf_, st_.s);
        if (cudaMemcpyAsync(out, d_out, (size_t) n * kernels::kLogprobStride * 4, cudaMemcpyDeviceToHost, st_.s) != cudaSuccess ||
            cudaStreamSynchronize(st_.s) != cudaSuccess)
            return {Status::Error, "inkling2: the logprobs kernel failed"};
        return {};
    }

private:
    Model& m_;
    const Hparams& h_;
    State st_;
    int last_T_ = 0;
    int32_t h_picks_[kWindow] = {};
    Prof prof_;
    kernels::SamplerParams sampling_{};
    bool bias_on_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr, *lp_buf_ = nullptr;
    int32_t* d_hist_ = nullptr;
    int hist_len_ = 0;
    bool device_floats(float*& dst, const float* src) {
        if (!dst) {
            if (cudaMalloc((void**) &dst, (size_t) h_.n_vocab * 4) != cudaSuccess) return false;
            st_.owned.push_back(dst);
        }
        return cudaMemcpy(dst, src, (size_t) h_.n_vocab * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool dec_experts_ = std::getenv("HALO_INK_DEC_EXPERTS") && std::atoi(std::getenv("HALO_INK_DEC_EXPERTS")) != 0;   // opt-in: slower than MMQ so far

    template <class T>
    T* alloc(size_t n, std::string& err) {
        void* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) {
            (void) cudaGetLastError();
            err = "inkling2: out of device memory for the session";
            return nullptr;
        }
        st_.owned.push_back(p);
        return (T*) p;
    }
    bool forward(const int32_t* tokens, int T, int64_t pos, bool all_logits, std::string& err, bool dev_pos = false);
    void q8(const Lin& l, const float* x, float* y, int T);
    void moe(const Layer& L, const float* x, float* out, int T);
    void attention(int visit, const Layer& L, int T, int64_t pos, bool dev_pos);
};

bool Session::init(std::string& err) {
    if (cudaStreamCreateWithFlags(&st_.s, cudaStreamNonBlocking) != cudaSuccess) {
        err = "inkling2: no CUDA stream";
        return false;
    }
    const int D = h_.n_embd, KV = h_.n_kv * h_.head_dim, AQ = h_.n_head * h_.head_dim, R = kChunk * h_.n_used;
    st_.kc.resize((size_t) h_.n_visit);
    st_.vc.resize((size_t) h_.n_visit);
    st_.ring.resize((size_t) h_.n_visit);
    for (int v = 0; v < h_.n_visit; ++v) {
        const bool swa = h_.is_swa[(size_t) v] != 0;
        const int64_t cells = swa ? std::min<int64_t>(kRing, m_.max_ctx_) : m_.max_ctx_;
        st_.ring[(size_t) v] = swa ? cells : 0;
        if (!(st_.kc[(size_t) v] = alloc<uint16_t>((size_t) cells * KV, err))) return false;
        if (!(st_.vc[(size_t) v] = alloc<uint16_t>((size_t) cells * KV, err))) return false;
    }
    st_.conv_floats = (int64_t) (2 * KV + 2 * D) * (h_.conv - 1);
    if (!(st_.conv_state = alloc<float>((size_t) st_.conv_floats * h_.n_visit, err))) return false;
    cudaMemset(st_.conv_state, 0, (size_t) st_.conv_floats * h_.n_visit * 4);
    const size_t TD = (size_t) kChunk * D;
    for (float** p : {&st_.cur, &st_.inp_sa, &st_.prelude_cur, &st_.bank_last, &st_.x, &st_.y, &st_.moe_out})
        if (!(*p = alloc<float>(TD, err))) return false;
    for (int j = 0; j < 3; ++j)
        if (!(st_.in3[j] = alloc<float>(TD, err))) return false;
    for (int j = 0; j < 4; ++j)
        if (!(st_.msg[j] = alloc<float>(TD, err))) return false;
    for (int j = 0; j < 10; ++j)
        if (!(st_.blocks[j] = alloc<float>(TD, err))) return false;
    const size_t TQ = (size_t) kChunk * AQ, TK = (size_t) kChunk * KV;
    if (!(st_.q = alloc<float>(TQ, err)) || !(st_.g = alloc<float>(TQ, err)) || !(st_.att = alloc<float>(TQ, err)) ||
        !(st_.k = alloc<float>(TK, err)) || !(st_.v = alloc<float>(TK, err)) ||
        !(st_.r = alloc<float>((size_t) kChunk * h_.rel_len * h_.n_head, err)) ||
        !(st_.sink = alloc<float>((size_t) kChunk * h_.n_head, err)) ||
        !(st_.kdyn = alloc<float>((size_t) kChunk * h_.conv * h_.n_kv, err)) ||
        !(st_.vdyn = alloc<float>((size_t) kChunk * h_.conv * h_.n_kv, err)) || !(st_.tau = alloc<float>(kChunk, err)) ||
        !(st_.rlogits = alloc<float>((size_t) kChunk * h_.n_expert, err)) ||
        !(st_.rweight = alloc<float>((size_t) kChunk * h_.n_used, err)) || !(st_.gu = alloc<float>((size_t) R * 2 * h_.n_ff, err)) ||
        !(st_.h = alloc<float>((size_t) R * h_.n_ff, err)) || !(st_.dn = alloc<float>((size_t) R * D, err)) ||
        !(st_.sel = alloc<int32_t>((size_t) R, err)) || !(st_.bounds = alloc<int32_t>((size_t) h_.n_expert + 1, err)) ||
        !(st_.row_of = alloc<int32_t>((size_t) R, err)) || !(st_.src_tok = alloc<int32_t>((size_t) R, err)) ||
        !(st_.expert_of = alloc<int32_t>((size_t) R, err)) || !(st_.iota = alloc<int32_t>((size_t) R, err)) ||
        !(st_.tok = alloc<int32_t>(kChunk, err)) || !(st_.picks = alloc<int32_t>(kWindow, err)) ||
        !(st_.pairs = alloc<int32_t>((size_t) 2 * (kChunk + 1), err)))
        return false;
    mmq::iota(st_.iota, R, st_.s);
    {   // pairs[2n] = 0, pairs[2n + 1] = n: the bounds of a one-matrix product of n rows
        std::vector<int32_t> pr((size_t) 2 * (kChunk + 1));
        for (int n = 0; n <= kChunk; ++n) pr[(size_t) 2 * n] = 0, pr[(size_t) 2 * n + 1] = n;
        cudaMemcpy(st_.pairs, pr.data(), pr.size() * 4, cudaMemcpyHostToDevice);
    }
    const size_t xq_bytes = std::max(mmq::q8_bytes(R, std::max<int64_t>(D, AQ)), mmq::q8_bytes(R, h_.n_ff)) + (1 << 20);
    if (!(st_.xq = alloc<uint8_t>(xq_bytes, err)) || !(st_.xq2 = alloc<uint8_t>(xq_bytes, err))) return false;
    if (!(st_.logits_pad = alloc<float>((size_t) kWindow * h_.n_vocab_padded, err)) ||
        !(st_.logits = alloc<float>((size_t) kWindow * h_.n_vocab, err)))
        return false;
    if (!(st_.d_pos = alloc<int64_t>(1, err))) return false;
    if (cudaHostAlloc((void**) &st_.h_pos, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &st_.h_tok, sizeof(int32_t) * kChunk, cudaHostAllocDefault) != cudaSuccess) {
        err = "inkling2: out of page-locked memory";
        return false;
    }
    k::reserve(kChunk);
    if (const char* g = std::getenv("HALO_INK_GRAPH"); g && std::atoi(g) == 0) st_.graph_state = -1;
    st_.attn_ws_bytes = k::ink_attention_workspace(kChunk, m_.max_ctx_);
    if (!(st_.attn_ws = alloc<uint8_t>(st_.attn_ws_bytes, err))) return false;
    if (cudaStreamSynchronize(st_.s) != cudaSuccess) {
        err = std::string("inkling2: session setup failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

// y [T][rows] = W x for a Q8_0 matrix: native mmvq for a few rows, MMQ otherwise
void Session::q8(const Lin& l, const float* x, float* y, int T) {
    cudaStream_t s = st_.s;
    if (T <= kMmvqRows && kernels::native_mmvq_supported(l.type)) {
        kernels::native_quantize_q8_1(x, st_.xq, (int) l.cols, T, s);
        kernels::native_mmvq(l.type, l.w, st_.xq, y, (int) l.cols, (int) l.rows, T, s);
        return;
    }
    mmq::quantize(x, nullptr, st_.xq, l.type, l.cols, l.cols, T, s);
    mmq::Product p;
    p.w = l.w;
    p.type = l.type;
    p.w_rows = l.rows;
    p.w_cols = l.cols;
    p.expert_bytes = mmq::matrix_bytes(l.type, l.rows, l.cols);
    p.n = 1;
    p.xq = st_.xq;
    p.bounds = st_.pairs + 2 * T;
    p.ids = st_.iota;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = l.rows;
    m_.mmq_ctx->run(p, s);
}

// out [T][D] = the layer's experts of x [T][D] (normed): route, group by expert, two grouped products, combine
void Session::moe(const Layer& L, const float* x, float* out, int T) {
    cudaStream_t s = st_.s;
    const int D = h_.n_embd, E = h_.n_expert, K = h_.n_used, F = h_.n_ff, R = T * K;
    prof_.mark(5, s);
    f32_matvec(L.router, x, st_.rlogits, T, D, E, s);
    k::moe_route(st_.rlogits, L.probs_b, T, E, K, h_.w_scale, st_.sel, st_.rweight, s);
    if (T <= kDecodeExperts && dec_experts_) {   // a few rows: the picked experts' rows read once, no grouping
        prof_.mark(6, s);
        k::moe_dec_gu(L.ex.gu, L.ex.gu_bytes, L.ex.type, st_.sel, x, L.ex.gu_scale, st_.h, T, K, D, F, h_.situ_beta, h_.situ_lbeta, s);
        k::moe_dec_down(L.ex.down, L.ex.d_bytes, L.ex.type, st_.sel, st_.rweight, L.ex.d_scale, st_.h, out, T, K, D, F, s);
        return;
    }
    k::moe_plan(st_.sel, T, K, E, st_.bounds, st_.row_of, st_.src_tok, st_.expert_of, s);
    prof_.mark(6, s);
    mmq::quantize(x, st_.src_tok, st_.xq, L.ex.type, D, D, R, s);
    mmq::Product p;
    p.w = L.ex.gu;
    p.type = L.ex.type;
    p.w_rows = 2 * F;
    p.w_cols = D;
    p.expert_bytes = L.ex.gu_bytes;
    p.n = E;
    p.xq = st_.xq;
    p.bounds = st_.bounds;
    p.ids = st_.iota;
    p.total_rows = R;
    p.max_rows = R;
    p.dst = st_.gu;
    p.ld_dst = 2 * F;
    m_.mmq_ctx->run(p, s);
    k::moe_situ(st_.gu, st_.expert_of, L.ex.gu_scale, st_.h, R, F, h_.situ_beta, h_.situ_lbeta, s);
    mmq::quantize(st_.h, nullptr, st_.xq2, L.ex.type, F, F, R, s);
    p.w = L.ex.down;
    p.w_rows = D;
    p.w_cols = F;
    p.expert_bytes = L.ex.d_bytes;
    p.xq = st_.xq2;
    p.dst = st_.dn;
    p.ld_dst = D;
    m_.mmq_ctx->run(p, s);
    k::moe_combine(st_.dn, st_.row_of, st_.sel, st_.rweight, L.ex.d_scale, out, T, K, D, s);
}

// the visit's attention from st_.in3 (normed inputs q, k, v) into st_.y [T][D] (after wo and the post convolution)
void Session::attention(int visit, const Layer& L, int T, int64_t pos, bool dev_pos) {
    const int64_t* dp = dev_pos ? st_.d_pos : nullptr;
    cudaStream_t s = st_.s;
    const int D = h_.n_embd, KV = h_.n_kv * h_.head_dim, AQ = h_.n_head * h_.head_dim;
    const float *iq = st_.in3[0], *ik = st_.in3[1], *iv = st_.in3[2];
    prof_.mark(2, s);
    float* cs = st_.conv_state + (size_t) visit * st_.conv_floats;
    const int64_t cw = h_.conv - 1;
    float *st_k = cs, *st_v = cs + (size_t) KV * cw, *st_ao = cs + (size_t) 2 * KV * cw;
    q8(L.q, iq, st_.q, T);
    q8(L.k, ik, st_.k, T);
    q8(L.v, iv, st_.v, T);
    q8(L.rel, iq, st_.r, T);
    q8(L.gate, iq, st_.g, T);
    f32_matvec(L.sink, iq, st_.sink, T, D, h_.n_head, s);
    if (!L.swa) {
        f32_matvec(L.k_cp, ik, st_.kdyn, T, D, h_.conv * h_.n_kv, s);
        f32_matvec(L.v_cp, iv, st_.vdyn, T, D, h_.conv * h_.n_kv, s);
    }
    prof_.mark(4, s);
    k::short_conv(st_.k, T, KV, L.k_conv, L.swa ? nullptr : st_.kdyn, h_.n_kv, st_k, s);
    k::short_conv(st_.v, T, KV, L.v_conv, L.swa ? nullptr : st_.vdyn, h_.n_kv, st_v, s);
    k::clip_rows(st_.q, L.q_clip, h_.n_head, (int64_t) T * h_.n_head, h_.head_dim, h_.qk_clip, s);
    k::clip_rows(st_.k, nullptr, 1, (int64_t) T * h_.n_kv, h_.head_dim, h_.qk_clip, s);
    k::rms_rows(st_.v, nullptr, st_.v, (int64_t) T * h_.n_kv, h_.head_dim, h_.eps, s);
    k::clip_rows(st_.r, L.rel_clip, h_.n_head, (int64_t) T * h_.n_head, h_.rel_len, h_.rel_clip, s);
    if (!L.swa) {
        k::log_tau(pos, dp, T, h_.log_floor, h_.log_alpha, st_.tau, s);
        k::scale_heads(st_.q, st_.tau, T, AQ, s);
        k::scale_heads(st_.r, st_.tau, T, h_.rel_len * h_.n_head, s);
    }
    const int64_t ring = st_.ring[(size_t) visit];
    k::kv_store(st_.k, st_.v, st_.kc[(size_t) visit], st_.vc[(size_t) visit], pos, dp, T, ring, s);
    k::AttnArgs a;
    a.q = st_.q;
    a.r = st_.r;
    a.sink = st_.sink;
    a.relb = L.rel_bias;
    a.kc = st_.kc[(size_t) visit];
    a.vc = st_.vc[(size_t) visit];
    a.ring = ring;
    a.pos0 = pos;
    a.pos_dev = dp;
    a.max_keys = m_.max_ctx_;
    a.T = T;
    a.window = L.swa ? h_.swa : 0;
    a.extent = L.swa ? h_.swa : h_.rel_extent;
    a.scale = 1.0f / std::sqrt((float) h_.head_dim);
    a.out = st_.att;
    a.workspace = st_.attn_ws;
    a.workspace_bytes = st_.attn_ws_bytes;
    prof_.mark(3, s);
    k::ink_attention(a, s);
    prof_.mark(4, s);
    k::sigmoid_gate(st_.att, st_.g, (int64_t) T * AQ, s);
    prof_.mark(2, s);
    q8(L.o, st_.att, st_.y, T);
    prof_.mark(4, s);
    k::short_conv(st_.y, T, D, L.post_conv, nullptr, 1, st_ao, s);
}

// The forward of T tokens at positions pos..: K/V and convolution state of every visit advance; the logits of the last
// row (or of every row with all_logits) land in st_.logits.
bool Session::forward(const int32_t* tokens, int T, int64_t pos, bool all_logits, std::string& err, bool dev_pos) {
    cudaStream_t s = st_.s;
    const int D = h_.n_embd;
    const int64_t TD = (int64_t) T * D;
    const Weights& W = m_.w;
    prof_.mark(0, s);
    if (!dev_pos && pos == 0) cudaMemsetAsync(st_.conv_state, 0, (size_t) st_.conv_floats * h_.n_visit * 4, s);   // a new sequence
    if (!dev_pos) cudaMemcpyAsync(st_.tok, tokens, (size_t) T * 4, cudaMemcpyHostToDevice, s);
    kernels::iq_embed_rows(W.embd_type, W.embd, W.embd_row_bytes, st_.tok, T, D, st_.cur, s);
    k::rms_rows(st_.cur, W.embd_norm, st_.cur, T, D, h_.eps, s);

    // blocks: indexes into st_.blocks; the list a mix reads is `blocks` then cur
    std::vector<int> blocks, prelude_blocks, bank;
    int next_block = 0;
    const int n_core = h_.core_end - h_.core_start;
    auto states = [&](const std::vector<int>& bl, const float* last, std::vector<const float*>& out) {
        out.clear();
        for (int b : bl) out.push_back(st_.blocks[b]);
        out.push_back(last);
    };
    std::vector<const float*> sv;
    for (int v = 0; v < h_.n_visit; ++v) {
        const int ip = h_.phys(v);
        const Layer& L = W.layers[(size_t) ip];
        prof_.mark(1, s);
        const bool core = h_.in_core(v);
        const int pass = h_.pass(v);
        bool has_msg = false;
        if (core && ip == h_.core_start) {
            if (pass == 0) {
                prelude_blocks = blocks;
                cudaMemcpyAsync(st_.prelude_cur, st_.cur, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
            } else {
                // the previous pass's blocks after the prelude's and the pass's first (the prelude state again)
                bank.assign(blocks.begin() + (long) prelude_blocks.size() + 1, blocks.end());
                cudaMemcpyAsync(st_.bank_last, st_.cur, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
                blocks = prelude_blocks;
                cudaMemcpyAsync(st_.cur, st_.prelude_cur, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
            }
        }
        if (core && pass > 0) {
            has_msg = true;
            const int64_t row = 2 * (ip - h_.core_start);
            states(bank, st_.bank_last, sv);
            // messages: 0 attention (q), 1 k, 2 v, 3 mlp
            k::depth_mix(sv.data(), (int) sv.size(), W.loop_res_query + row * D, 1, st_.msg[0], T, D, s);
            k::depth_mix(sv.data(), (int) sv.size(), W.loop_res_query + (row + 1) * D, 1, st_.msg[3], T, D, s);
            k::depth_mix(sv.data(), (int) sv.size(), W.loop_kv_res_query + row * D, 1, st_.msg[1], T, D, s);
            k::depth_mix(sv.data(), (int) sv.size(), W.loop_kv_res_query + (row + 1) * D, 1, st_.msg[2], T, D, s);
        }
        cudaMemcpyAsync(st_.inp_sa, st_.cur, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
        // attention inputs
        if (L.attn_res_query) {
            states(blocks, st_.cur, sv);
            for (int j = 0; j < 3; ++j) {
                if (L.swa && j > 0) {
                    cudaMemcpyAsync(st_.in3[j], st_.in3[0], (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
                    continue;
                }
                k::depth_mix(sv.data(), (int) sv.size(), L.attn_res_query + (int64_t) j * D, 1, st_.in3[j], T, D, s);
            }
        } else {
            for (int j = 0; j < 3; ++j) cudaMemcpyAsync(st_.in3[j], st_.cur, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
        }
        if (has_msg)
            for (int j = 0; j < 3; ++j)
                k::loop_control(st_.in3[j], st_.msg[j], L.loop_down[j], L.loop_gain[j], L.loop_forget[j], st_.in3[j], T, D,
                                h_.gate_rank, s);
        if (L.swa) {
            for (int j = 0; j < 3; ++j) k::rms_rows(st_.in3[j], L.attn_norm, st_.in3[j], T, D, h_.eps, s);
        } else {
            k::rms_rows(st_.in3[0], L.q_inp_norm, st_.in3[0], T, D, h_.eps, s);
            k::rms_rows(st_.in3[1], L.k_inp_norm, st_.in3[1], T, D, h_.eps, s);
            k::rms_rows(st_.in3[2], L.v_inp_norm, st_.in3[2], T, D, h_.eps, s);
        }
        attention(v, L, T, pos, dev_pos);
        // the first layer of a block starts a new residual stream; the stream before it joins the blocks
        if (ip % h_.res_block == 0) {
            if (next_block >= 10) {
                err = "inkling2: more residual blocks than the forward keeps";
                return false;
            }
            cudaMemcpyAsync(st_.blocks[next_block], st_.inp_sa, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
            blocks.push_back(next_block++);
            cudaMemcpyAsync(st_.cur, st_.y, (size_t) TD * 4, cudaMemcpyDeviceToDevice, s);
        } else {
            k::add_rows(st_.cur, st_.y, TD, s);
        }
        // the experts
        prof_.mark(1, s);
        states(blocks, st_.cur, sv);
        k::depth_mix(sv.data(), (int) sv.size(), L.ffn_res_query, 1, st_.x, T, D, s);
        if (has_msg) k::loop_control(st_.x, st_.msg[3], L.loop_down[3], L.loop_gain[3], L.loop_forget[3], st_.x, T, D, h_.gate_rank, s);
        k::rms_rows(st_.x, L.ffn_norm, st_.x, T, D, h_.eps, s);
        moe(L, st_.x, st_.moe_out, T);
        prof_.mark(4, s);
        float* cs = st_.conv_state + (size_t) v * st_.conv_floats;
        const int64_t KV = (int64_t) h_.n_kv * h_.head_dim, cw = h_.conv - 1;
        k::short_conv(st_.moe_out, T, D, L.ffn_post_conv, nullptr, 1, cs + (size_t) (2 * KV + D) * cw, s);
        k::add_rows(st_.cur, st_.moe_out, TD, s);
        (void) n_core;
    }
    // the head: mix, norm, the rows asked for
    prof_.mark(7, s);
    states(blocks, st_.cur, sv);
    k::depth_mix(sv.data(), (int) sv.size(), W.out_res_query, 1, st_.x, T, D, s);
    const int r0 = all_logits ? 0 : T - 1, rows = all_logits ? T : 1;
    k::rms_rows(st_.x + (size_t) r0 * D, W.out_norm, st_.x + (size_t) r0 * D, rows, D, h_.eps, s);
    q8(W.head, st_.x + (size_t) r0 * D, st_.logits_pad, rows);
    k::softcap(st_.logits_pad, (int64_t) rows * h_.n_vocab_padded, h_.softcap, s);
    const int64_t n = (int64_t) rows * h_.n_vocab;
    copy_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(st_.logits_pad, h_.n_vocab_padded, st_.logits, h_.n_vocab, rows);
    last_T_ = rows;
    prof_.end(s);
    if (cudaGetLastError() != cudaSuccess) {
        err = std::string("inkling2: a launch failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

Result Session::decode_window(const model::DecodeRequest& r) {
    if (r.n_tokens < 1 || r.n_tokens > kWindow) return {Status::Error, "inkling2: a decode window of " + std::to_string(r.n_tokens) + " rows"};
    if (r.pos + r.n_tokens > m_.max_ctx_) return {Status::Error, "inkling2: the window passes the context"};
    std::string err;
    if (r.n_tokens > 1) {   // a verify window: every row's logits, no graph
        if (!forward(r.tokens, r.n_tokens, r.pos, true, err)) return {Status::Error, err};
    } else if (st_.graph_state >= 1) {   // the captured one-row forward: token and position through device memory
        st_.h_tok[0] = r.tokens[0];
        *st_.h_pos = r.pos;
        cudaMemcpyAsync(st_.tok, st_.h_tok, 4, cudaMemcpyHostToDevice, st_.s);
        cudaMemcpyAsync(st_.d_pos, st_.h_pos, 8, cudaMemcpyHostToDevice, st_.s);
        if (st_.graph_state == 1) {
            cudaGraph_t g = nullptr;
            prof_.capturing = true;
            bool ok = cudaStreamBeginCapture(st_.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            ok = ok && forward(r.tokens, 1, r.pos, true, err, true);
            ok = cudaStreamEndCapture(st_.s, &g) == cudaSuccess && ok;
            prof_.capturing = false;
            ok = ok && cudaGraphInstantiate(&st_.graph, g, 0) == cudaSuccess;
            if (g) cudaGraphDestroy(g);
            if (!ok) {
                (void) cudaGetLastError();
                halo::log::info("halo inkling2", "the decode graph did not capture (%s); decoding without it", err.c_str());
                st_.graph_state = -1;
                err.clear();
            } else {
                st_.graph_state = 2;
            }
        }
        if (st_.graph_state == 2 && cudaGraphLaunch(st_.graph, st_.s) != cudaSuccess)
            return {Status::Error, std::string("inkling2: the decode graph failed: ") + cudaGetErrorString(cudaGetLastError())};
        if (st_.graph_state == -1 && !forward(r.tokens, r.n_tokens, r.pos, true, err)) return {Status::Error, err};
        last_T_ = 1;
    } else {
        if (!forward(r.tokens, r.n_tokens, r.pos, true, err)) return {Status::Error, err};
        if (st_.graph_state == 0) st_.graph_state = 1;   // warmed up: the next window captures
    }
    kernels::SamplerParams sp;
    sp.greedy = true;
    if (r.sampling) sp = *r.sampling;
    sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
    sampling_ = sp;
    if (r.history && r.history_len > 0) {   // the penalties' history of the row
        if (!d_hist_ && cudaMalloc((void**) &d_hist_, (size_t) 4096 * 4) == cudaSuccess) st_.owned.push_back(d_hist_);
        hist_len_ = std::min(r.history_len, 4096);
        if (d_hist_) cudaMemcpyAsync(d_hist_, r.history + (r.history_len - hist_len_), (size_t) hist_len_ * 4, cudaMemcpyHostToDevice, st_.s);
    } else {
        hist_len_ = 0;
    }
    const int T = r.n_tokens;
    kernels::sample_tokens(st_.logits, T, (int) h_.n_vocab, hist_len_ && d_hist_ && T == 1 ? d_hist_ : nullptr, T == 1 ? hist_len_ : 0, sp,
                           st_.picks, st_.s);
    if (r.logits) cudaMemcpyAsync(r.logits, st_.logits + (size_t) (T - 1) * h_.n_vocab, (size_t) h_.n_vocab * 4, cudaMemcpyDeviceToHost, st_.s);
    cudaMemcpyAsync(h_picks_, st_.picks, (size_t) T * 4, cudaMemcpyDeviceToHost, st_.s);
    if (cudaStreamSynchronize(st_.s) != cudaSuccess) return {Status::Error, std::string("inkling2: ") + cudaGetErrorString(cudaGetLastError())};
    if (r.row_tokens) std::copy(h_picks_, h_picks_ + T, r.row_tokens);
    if (r.next_token) *r.next_token = h_picks_[T - 1];
    return {};
}

Result Session::prefill_chunk(const model::PrefillRequest& r) {
    if (r.pos + r.n_tokens > m_.max_ctx_) return {Status::Error, "inkling2: the prompt passes the context"};
    for (int at = 0; at < r.n_tokens;) {
        const int n = std::min(kChunk, r.n_tokens - at);
        std::string err;
        if (!forward(r.tokens + at, n, r.pos + at, false, err)) return {Status::Error, err};
        if (cudaStreamSynchronize(st_.s) != cudaSuccess) return {Status::Error, std::string("inkling2: ") + cudaGetErrorString(cudaGetLastError())};
        at += n;
        if (r.progress) r.progress(at);
        if (at < r.n_tokens && r.stop && r.stop()) break;
    }
    return {};
}

Result Model::make_session(std::unique_ptr<model::ModelSession>& out) {
    auto s = std::make_unique<Session>(*this);
    std::string err;
    if (!s->init(err)) return {Status::Error, err};
    out = std::move(s);
    return {};
}

uint64_t u(const GgufFile& gg, const std::string& k, bool& ok) {
    const MetaValue* v = gg.get("inkling2." + k);
    if (!v || !v->is_num()) {
        ok = false;
        return 0;
    }
    return v->u;
}
float f(const GgufFile& gg, const std::string& k, bool& ok) {
    const MetaValue* v = gg.get("inkling2." + k);
    if (!v || !v->is_num()) {
        ok = false;
        return 0;
    }
    return (float) v->num();
}
// a per-visit array key's first value, or the scalar
uint64_t first(const GgufFile& gg, const std::string& k, bool& ok) {
    const MetaValue* v = gg.get("inkling2." + k);
    if (!v) {
        ok = false;
        return 0;
    }
    if (v->type == MetaType::ARRAY) return v->items.empty() ? 0 : v->items.front().u;
    return v->u;
}

}  // namespace

bool read_hparams(const GgufFile& gg, Hparams& h, std::string& err) {
    const MetaValue* a = gg.get("general.architecture");
    if (!a || a->s != "inkling2") {
        err = "inkling2: the file's architecture is not inkling2";
        return false;
    }
    bool ok = true;
    h.n_embd = (int) u(gg, "embedding_length", ok);
    h.n_visit = (int) u(gg, "block_count", ok);
    h.n_head = (int) first(gg, "attention.head_count", ok);
    h.n_kv = (int) first(gg, "attention.head_count_kv", ok);
    h.head_dim = (int) u(gg, "attention.key_length", ok);
    h.n_pass = (int) u(gg, "num_loops", ok);
    h.core_start = (int) u(gg, "loop.core_start", ok);
    h.core_end = (int) u(gg, "loop.core_end", ok);
    h.gate_rank = (int) u(gg, "loop.gate_rank", ok);
    h.n_expert = (int) u(gg, "expert_count", ok);
    h.n_used = (int) u(gg, "expert_used_count", ok);
    h.n_ff = (int) first(gg, "expert_feed_forward_length", ok);
    h.n_vocab_padded = (int64_t) u(gg, "vocab_size", ok);
    h.n_vocab = (int64_t) u(gg, "vocab_size_unpadded", ok);
    h.context_length = (int64_t) u(gg, "context_length", ok);
    h.swa = (int) u(gg, "attention.sliding_window", ok);
    h.rel_len = (int) u(gg, "attention.relative_length", ok);
    h.rel_extent = (int) u(gg, "attention.relative_extent", ok);
    h.conv = (int) u(gg, "shortconv.l_cache", ok);
    h.res_block = (int) u(gg, "attn_res.block_size", ok);
    h.eps = f(gg, "attention.layer_norm_rms_epsilon", ok);
    h.qk_clip = f(gg, "attention.qk_clip", ok);
    h.rel_clip = f(gg, "attention.relative_clip", ok);
    h.log_floor = f(gg, "attention.log_scale_floor", ok);
    h.log_alpha = f(gg, "attention.log_scale_alpha", ok);
    h.situ_beta = f(gg, "activation.situ_beta", ok);
    h.situ_lbeta = f(gg, "activation.situ_linear_beta", ok);
    h.w_scale = f(gg, "expert_weights_scale", ok);
    h.softcap = f(gg, "final_logit_softcapping", ok);
    if (!ok) {
        err = "inkling2: the file lacks an inkling2.* key the model needs";
        return false;
    }
    const MetaValue* pat = gg.get("inkling2.attention.sliding_window_pattern");
    if (!pat || pat->type != MetaType::ARRAY || (int) pat->items.size() != h.n_visit) {
        err = "inkling2: the sliding-window pattern is not one flag per visit";
        return false;
    }
    for (const MetaValue& x : pat->items) h.is_swa.push_back(x.u ? 1 : 0);
    const int n_core = h.core_end - h.core_start;
    h.n_phys = h.n_pass > 1 ? h.n_visit - n_core * (h.n_pass - 1) : h.n_visit;
    if (h.n_embd != 2304 || h.head_dim != 128 || h.n_head % h.n_kv || h.n_used > 16 || h.conv != 4 || h.res_block <= 0 ||
        h.core_start % h.res_block || n_core % h.res_block || h.n_phys <= 0 || h.n_vocab > h.n_vocab_padded) {
        err = "inkling2: a shape this engine does not run (width 2304, head 128, conv 4, whole blocks in the core)";
        return false;
    }
    // every visit of a stored layer has the same kind
    for (int v = 0; v < h.n_visit; ++v)
        for (int w = 0; w < v; ++w)
            if (h.phys(v) == h.phys(w) && h.is_swa[(size_t) v] != h.is_swa[(size_t) w]) {
                err = "inkling2: the visits of a stored layer disagree on sliding-window";
                return false;
            }
    return true;
}

std::unique_ptr<model::LoadedModel> load_model(const model::LoadRequest& req, std::string& err) {
    auto m = std::make_unique<Model>();
    try {
        if (!m->load(req, err)) return nullptr;
    } catch (const std::exception& e) {
        err = std::string("inkling2: ") + e.what();
        return nullptr;
    }
    return m;
}

}  // namespace halo::models::inkling2
