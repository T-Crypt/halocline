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
    const uint16_t* kc = nullptr;    ///< BF16 [cells][8][128]
    const uint16_t* vc = nullptr;
    int64_t ring = 0;                ///< 0: linear (cell = position); else cell = position % ring
    int64_t pos0 = 0;                ///< position of row 0; the cache holds every position <= pos0 + T - 1
    int T = 0;
    int64_t window = 0;              ///< 0 full; W: keys pos_t - W + 1 .. pos_t
    int extent = 1024;               ///< relative bias past this distance is 0
    float scale = 0.0883883476f;     ///< 1 / sqrt(128)
    float* out = nullptr;            ///< [T][24][128]
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
};

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

/// tau[t] = 1 + alpha log(max(1, (pos0 + t + 1) / floor)).
void log_tau(int64_t pos0, int T, float floor, float alpha, float* tau, void* stream);
/// x[t][*] *= tau[t].
void scale_heads(float* x, const float* tau, int T, int per_row, void* stream);
/// o *= sigmoid(g).
void sigmoid_gate(float* o, const float* g, int64_t n, void* stream);
/// x = cap tanh(x / cap).
void softcap(float* x, int64_t n, float cap, void* stream);
/// K and V rows [T][8][128] into the BF16 caches at the cells of positions pos0.. (ring as in AttnArgs).
void kv_store(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int64_t pos0, int T, int64_t ring, void* stream);
/// x += y.
void add_rows(float* x, const float* y, int64_t n, void* stream);

}  // namespace halo::models::inkling2::k
