// src/models/inkling2/kernels/ink_ops.hpp - the non-GEMM kernels of Inkling 2 Mini on sm_89 (the products run on the
// engine's MMQ and native mmvq). FP32 activations, row-major; T rows a call (1..1024); D = 2304; 24 query heads over 8
// KV heads (group 3) of 128. Every accumulation is FP32. The math follows llama.cpp's inkling2 graph.
#pragma once
#include <cstddef>
#include <cstdint>

namespace halo::models::inkling2::k {

/// Per row of `width` values: x *= scale (row % scale_rows of scale [scale_rows][width], when given), then
/// x *= min(1, threshold / (||x|| + 1e-6)).
void clip_rows(float* x, const float* scale, int scale_rows, int64_t rows, int width, float threshold, void* stream);

/// out = x / sqrt(mean(x^2) + eps) * (w ? w : 1) per row of `width`; out may be x.
void rms_rows(const float* x, const float* w, float* out, int64_t rows, int width, float eps, void* stream);

/// Attention over the depth: per row t and query qi, s_j = rms(states[j][t]) (eps 1e-6), a_j = dot(query[qi], s_j)
/// / sqrt(D), p = softmax(a), out[t][qi] = sum_j p_j s_j. `states`: a host array of n_states (1..16) device pointers,
/// each [T][D]; query [n_query][D] (1..3); out [T][n_query][D] (distinct from every state).
void depth_mix(const float* const* states, int n_states, const float* query, int n_query, float* out, int T, int D,
               void* stream);

/// The looped core's message controller, per row: h = [rms(cur); rms(msg)] (eps 1e-6), z = w_down h (rank values),
/// g = tanh(w_gain z), f = tanh(w_forget z), out = (1 - f) cur + g msg (no FMA in the merge). w_down [rank][2D],
/// w_gain and w_forget [D][rank]; out may be cur.
void loop_control(const float* cur, const float* msg, const float* w_down, const float* w_gain, const float* w_forget,
                  float* out, int T, int D, int rank, void* stream);

/// The causal depthwise convolution of width 4 with a carried state (state [C][3], oldest first) and optional
/// input-dependent taps (dyn [T][n_heads][4]): x_out[t] = (x[t] + sum_k xx[t+k] w[c][k] + sum_lag xx[t+3-lag]
/// dyn[t][head][lag]) / sqrt((dyn ? 2 : 1) * 4 + 1), xx = state then x; the state then holds the last 3 inputs.
void short_conv(float* x, int T, int C, const float* w, const float* dyn, int n_heads, float* state, void* stream);

struct AttnArgs {
    const float* q = nullptr;        ///< [T][24][128]
    const float* r = nullptr;        ///< [T][24][16]
    const float* sink = nullptr;     ///< [T][24]
    const float* relb = nullptr;     ///< [extent][16]
    const void* kc = nullptr;        ///< the K cache in kfmt (KvFmt), cell c head g at row c * 8 + g
    const void* vc = nullptr;
    const float* ks = nullptr;       ///< int8: a scale per row [cells][8]; q4: per 32 values [cells][8][4]
    const float* vs = nullptr;
    int kfmt = 0, vfmt = 0;          ///< KvFmt; a q4 K means q arrives rotated (rotate_rows), a q4 V leaves out rotated
    int64_t ring = 0;                ///< 0: linear (cell = position); else cell = position % ring
    int64_t pos0 = 0;                ///< position of row 0; the cache holds every position <= pos0 + T - 1
    const int64_t* pos_dev = nullptr;   ///< when given, row 0's position is *pos_dev (a CUDA graph replays one launch for
                                        ///< every position) and the splits are fixed by max_keys
    int64_t max_keys = 0;            ///< with pos_dev: the most keys a row may have
    int T = 0;
    int64_t window = 0;              ///< 0 full; W: keys pos_t - W + 1 .. pos_t
    int extent = 1024;               ///< relative bias past this distance is 0
    float scale = 0.0883883476f;     ///< 1 / sqrt(128)
    float* out = nullptr;            ///< [T][24][128]
    bool uniform = false;            ///< every row at position pos0 (a draft block reads the context before its anchor)
    const float* xk = nullptr;       ///< extra keys after the cache's, FP32 [xn][8][128] (a draft block's own), seen by
    const float* xv = nullptr;       ///< every row
    int xn = 0;
    bool no_bias = false;            ///< no relative bias and no sink (a drafter's plain attention)
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

/// K/V cell formats: BF16 [128] a row; int8 [128] + an fp32 scale; 4-bit after a 128-point Hadamard rotation, [64] bytes
/// (value 2i low nibble, 2i + 1 high, offset by 8) + an fp32 scale per 32 values (q4_0's rounding).
enum KvFmt { kKvBf16 = 0, kKvInt8 = 1, kKvQ4 = 2 };
/// Bytes of a row's codes and the floats of its scales in a format.
inline size_t kv_row_bytes(int fmt) { return fmt == kKvBf16 ? 256 : fmt == kKvInt8 ? 128 : 64; }
inline size_t kv_row_scales(int fmt) { return fmt == kKvBf16 ? 0 : fmt == kKvInt8 ? 1 : 4; }
/// x [rows][128] times the normalized 128-point Hadamard matrix (its own inverse), in place.
void rotate_rows(float* x, int64_t rows, void* stream);

/// Bytes of ink_attention's split-KV partials for up to T rows over up to n_keys_max keys.
size_t ink_attention_workspace(int T, int64_t n_keys_max);

/// Relative-bias attention with a sink: for row t, head h (KV head h / 3) and each allowed key j, s_j = scale
/// dot(q, k_j) + (dist < extent ? dot(r[t][h], relb[dist]) : 0), dist = pos_t - j; p_j = exp(s_j) / (exp(sink) +
/// sum exp(s)); out = sum p_j v_j. Split-KV (flash decoding) pass and a combine pass.
void ink_attention(const AttnArgs& a, void* stream);

/// p = sigmoid(logit); the k experts of largest p + bias (ties: lower index), in that order; weight = p / (sum of the
/// k p + 1e-20) * w_scale.
void moe_route(const float* logits, const float* bias, int T, int n_expert, int k, float w_scale, int32_t* sel,
               float* weight, void* stream);

/// The T * k picks sorted by expert, stable in (token, slot): bounds [n_expert + 1]; row_of [T * k] the sorted row of
/// pick t * k + slot; src_token and expert_of [T * k] per sorted row.
void moe_plan(const int32_t* sel, int T, int k, int n_expert, int32_t* bounds, int32_t* row_of, int32_t* src_token,
              int32_t* expert_of, void* stream);

/// h = beta tanh(g / beta) sigmoid(g) * lbeta tanh(u / lbeta) of gu rows [R][2F] (gate F then up F), g and u first
/// times gu_scale[expert_of[row]] when given.
void moe_situ(const float* gu, const int32_t* expert_of, const float* gu_scale, float* h, int R, int F, float beta,
              float lbeta, void* stream);

/// out[t] = sum_slot weight[t][slot] * d_scale[sel[t][slot]] * down[row_of[t * k + slot]] (d_scale optional).
void moe_combine(const float* down, const int32_t* row_of, const int32_t* sel, const float* weight, const float* d_scale,
                 float* out, int T, int k, int D, void* stream);

/// Decode-sized experts (T * k picks, T small): h[pick] = situ of the picked expert's gate/up rows of x[t] (with its
/// gu_scale), straight from the GGUF blocks (NVFP4 type 40 or Q8_0 type 8) in FP32. `gu`: the experts' [2F][D] matrices,
/// `gu_bytes` apart.
void moe_dec_gu(const void* gu, size_t gu_bytes, int type, const int32_t* sel, const float* x, const float* gu_scale,
                float* h, int T, int k, int D, int F, float beta, float lbeta, void* stream);
/// out[t] = sum_slot weight * d_scale[e] * (W_down[e] h[pick]) for the picks of moe_dec_gu; `down`: [D][F] matrices.
void moe_dec_down(const void* down, size_t d_bytes, int type, const int32_t* sel, const float* weight, const float* d_scale,
                  const float* h, float* out, int T, int k, int D, int F, void* stream);

/// NeoX rotary embedding of x [T][heads][128] at positions pos0 + t (theta: the base), in place.
void rope_neox(float* x, int T, int heads, int64_t pos0, float theta, void* stream);
/// h = silu(g) * u over n values.
void swiglu(const float* g, const float* u, float* h, int64_t n, void* stream);
/// out[t] = sum_j softmax(fuse[0..nj))[j] * feat[t][j] for features feat [T][nj][D]; fuse a device row of nj floats.
void mix_features(const float* feat, const float* fuse, float* out, int T, int nj, int D, void* stream);
/// The DSpark Markov step's input: x = [state (R) ; w1[prev] (R, BF16 rows of R) ; hidden (D) ; ctx (D)].
void markov_input(const float* state, const uint16_t* w1, const int32_t* prev, const float* hidden, const float* ctx, float* x,
                  int R, int D, void* stream);
/// After joint = J x + b (3R): state = cand + sigmoid(gate) (state - cand), latent = tanh(lat) (cand = tanh(joint[R..2R))).
void markov_update(const float* joint, float* state, float* latent, int R, void* stream);
/// The first maximum of logits [n] into *id and its softmax probability into *prob (one block).
void argmax_prob(const float* logits, int n, int32_t* id, float* prob, void* stream);
/// x = a + b over n values.
void add2(const float* a, const float* b, float* out, int64_t n, void* stream);
/// Per-row state history for decode windows: short_conv_hist is short_conv that also writes the state after every row
/// to hist [T][C][3]; conv_restore copies row r of a history back into the state.
void short_conv_hist(float* x, int T, int C, const float* w, const float* dyn, int n_heads, float* state, float* hist,
                     int64_t hist_row, void* stream);
/// Y[T][N] = X[T][K] . W[N][K]^T for a BF16 W (cuBLAS, FP32 accumulate; X rounded to BF16 on the way in).
void gemm_bf16(const uint16_t* W, const float* X, float* Y, int T, int N, int K, void* stream);

/// tau[t] = 1 + alpha log(max(1, (pos0 + t + 1) / floor)); pos_dev, when given, holds pos0.
void log_tau(int64_t pos0, const int64_t* pos_dev, int T, float floor, float alpha, float* tau, void* stream);
/// x[t][*] *= tau[t].
void scale_heads(float* x, const float* tau, int T, int per_row, void* stream);
/// o *= sigmoid(g).
void sigmoid_gate(float* o, const float* g, int64_t n, void* stream);
/// x = cap tanh(x / cap).
void softcap(float* x, int64_t n, float cap, void* stream);
/// K and V rows [T][8][128] into the BF16 caches at the cells of positions pos0.. (ring as in AttnArgs).
void kv_store(const float* k, const float* v, void* kc, void* vc, float* ks, float* vs, int kfmt, int vfmt, int64_t pos0,
              const int64_t* pos_dev, int T, int64_t ring, void* stream);
/// Sizes the kernels' own scratch for calls of up to T rows (before a CUDA graph capture, which cannot allocate).
void reserve(int T);
/// x += y.
void add_rows(float* x, const float* y, int64_t n, void* stream);

}  // namespace halo::models::inkling2::k
