// include/halo/dense/model.hpp - a dense decoder of hybrid layers: each layer a token mixer (gated delta-net or
// gated grouped-query attention) and a dense SwiGLU FFN, with offset RMS norms, an fp32 residual and an untied head.
//
// Shared code: a model folder reads its pack into `Geometry` and `Weights` (device pointers it owns) and gets
// sessions from `make_session`, which the shared session engine serves (parallel slots, conversation cache,
// drafting, grammar, logprobs, images).  Nothing here names a model; every width comes from the geometry.
//
// Products: decode windows (up to 8 rows) on q8_1 activations through the shared MMVQ (Q8_0, NVFP4); prompt chunks
// through llama.cpp's MMQ on int8 tensor cores.  An NVFP4 matrix's fp32 tensor scale is applied after its product.
#pragma once

#include "halo/core/model_file.hpp"
#include "halo/kernels/dense_ops.hpp"
#include "halo/kernels/gdn_window.hpp"
#include "halo/model/adapter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace halo::dense {

enum class Mixer { Gdn, Attention };

/// One projection: `rows` outputs of `cols` inputs, ggml blocks of `type` (Q8_0 8, NVFP4 40, F8_E4M3 44: one E4M3 code
/// per value, rows of `cols` bytes) on the device; `scale` an NVFP4 or F8 matrix's fp32 tensor scale (device), null for 1.
/// An NVFP4 product leaves the scale to the step after it; an F8 product applies it itself.  The head alone may be F8
/// with a scale per row (`row_scales`, fp32 [rows] on the device, `scale` then null): it runs on the window gemm only.
struct Linear {
    int type = -1;
    const void* w = nullptr;
    int64_t rows = 0, cols = 0;
    const float* scale = nullptr;
    const float* row_scales = nullptr;
};

struct Layer {
    Mixer kind = Mixer::Gdn;
    const float* attn_norm = nullptr;   ///< the mixer's input norm, offset folded (1 + w)
    const float* ffn_norm = nullptr;    ///< the FFN's
    // gated delta-net: value heads in the order where value head h meets key head h % k_heads
    Linear qkv, z, out;
    const uint16_t* w_ab = nullptr;     ///< bf16 [2 v_heads][n_embd]: the alpha rows, then the beta rows
    const float *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *gamma = nullptr;
    // attention: q gives [query | gate] per head
    Linear q, k, v, o;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // FFN
    Linear gate, up, down;
};

/// The drafter layer a pack may carry (multi-token prediction): fc(concat(norm_e(embed(next)), norm_h(hidden))),
/// one attention layer with its FFN, a final norm, the main head.
struct Mtp {
    bool present = false;
    Linear fc;
    const float *norm_e = nullptr, *norm_h = nullptr, *norm = nullptr;
    Layer layer;
};

/// A block drafter (DFlash 2): the target's rows at a few layers, projected and normed, are the context (K/V in a
/// sliding window) of a few non-causal attention layers that read a block [anchor, mask, ...] and give every slot's
/// candidates at once; a selector walks one path through them.  The head is the target's.
struct BlockDrafter {
    struct Layer {
        const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        const float *base_a = nullptr, *base_f = nullptr;   // the two convolutions' base kernels [2][2][n_embd]
        Linear q, k, v, o, gate, up, down, conv_a, conv_f;
    };
    bool present = false;
    int block = 0, mask_id = 0, conv_group = 0, rank = 0, top_k = 0, window = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    float theta = 0.0f, eps = 1e-6f;
    std::vector<int> target_layers;          ///< the target layers whose outputs it reads, in order
    std::vector<Linear> fc;                  ///< per target layer, its block of the feature projection
    const float *hidden_norm = nullptr, *norm = nullptr;
    std::vector<Layer> layers;
    const uint16_t *sel_h = nullptr, *pred = nullptr, *succ = nullptr;   // bf16 [rank][n_embd], [vocab][rank] x 2
};

struct Geometry {
    int n_layers = 0, n_embd = 0, n_ff = 0, n_vocab = 0;
    int n_head = 0, n_kv = 0, head_dim = 0;
    kernels::RopeSpec rope;
    int gdn_state = 0, gdn_k_heads = 0, gdn_v_heads = 0, gdn_conv = 0;
    kernels::GdnGate gdn_gate = kernels::GdnGate::Silu;
    float eps = 1e-6f;
    std::vector<Mixer> kinds;
    int64_t context_length = 0;          ///< the longest context the model supports
    int gdn_layers() const;
    int attn_layers() const;
    int conv_channels() const { return 2 * gdn_k_heads * gdn_state + gdn_v_heads * gdn_state; }
    int value_width() const { return gdn_v_heads * gdn_state; }
    /// Empty when the shared kernels take this geometry, else why not.
    std::string check() const;
};

struct Weights {
    std::vector<Layer> layers;
    const void* embd = nullptr;         ///< [n_vocab][n_embd] of embd_type, device-visible (may be mapped host memory)
    int embd_type = 30;                 ///< BF16, or the GGUF quant the pack stores (kernels::embed_type_supported)
    size_t embd_row_bytes = 0;          ///< one row's bytes (0: n_embd BF16 values)
    const float* out_norm = nullptr;
    Linear head;
    Mtp mtp;
    BlockDrafter dflash;
};

/// Which drafter a model's sessions run.
enum class Drafter { None, Mtp, Block };

/// --mtp-kv's value: 1 fp16 (the default, also for an empty value), 2 the attention layers' int8 codes, -1 for neither.
inline int mtp_kv_format(const std::string& v) {
    if (v.empty() || v == "fp16") return 1;
    return v == "int8" ? 2 : -1;
}

/// The cells a windowed drafter's ring holds: its window, plus the rows one pass writes before it reads them, and at
/// least one 32-cell sweep chunk of the decode kernel (a chunk's cells wrap at most once). A pass of up to kGqaMaxWindow
/// rows writes cells c0 .. c0 + T - 1 and its first row reads from c0 - window + 1, so the widest gap between a row's
/// oldest read and the highest write of the pass is window + T - 2; one more cell keeps every read off a cell that same
/// pass overwrote.
inline int64_t mtp_ring_cells(int64_t window) {
    const int64_t cells = window + kernels::kGqaMaxWindow - 1;
    return cells < kernels::kGqaCellChunk ? kernels::kGqaCellChunk : cells;
}

/// The ring's cells for a window against a context: mtp_ring_cells(window), or 0 (cells at their own positions, as
/// without a window) when the window reaches the context or the ring would hold it anyway.
inline int64_t mtp_ring_cap(int64_t max_context, int64_t window) {
    if (window <= 0 || max_context <= 0 || window >= max_context) return 0;
    const int64_t cells = mtp_ring_cells(window);
    return cells < max_context ? cells : 0;
}

struct Options {
    int64_t max_context = 0;            ///< cells each session holds
    int sessions = 1;
    int spec = 0;                       ///< rows a decode window may hold (0 or 1: one)
    bool draft = false;                 ///< run the pack's drafter
    Drafter drafter = Drafter::None;    ///< which one
    /// The MTP drafter's own K/V: 1 fp16 (the default, as every pack has run), 2 the attention layers' int8 codes
    /// (--mtp-kv int8: half the bytes, and the drafter's choice is only ever a draft). A third format, the main cache's
    /// k8v4, would be 3 here: the format reaches the kernels as the same KvInt8 struct the main cache uses.
    int mtp_format = 1;
    /// --mtp-window: cells of context the drafter's attention reads (0: every cell, the default). Above 0 the drafter's
    /// cells are kept in a ring of mtp_ring_cells(N), so the cache costs those cells rather than the context's.
    int64_t mtp_window = 0;
    uint64_t host_embedding_bytes = 0;  ///< the token embedding the adapter keeps in page-locked RAM (a row lookup per token)
    uint64_t gpu_weight_bytes = 0;      ///< the weights the adapter put on the main card (its "GiB on the GPU" load figure, no KV cache)
    bool vision = false;                ///< requests may carry images
    bool kv_int8 = false;               ///< the attention layers' K/V as int8 codes with a scale per 64 values
    /// K/V streaming: each attention layer keeps its first kv_resident cells on the card and the rest in page-locked RAM,
    /// read over the bus by every decode window and staged to the card for each prompt chunk (0, or at least max_context:
    /// all on the card)
    int64_t kv_resident = 0;
    core::FileRange mmproj;             ///< the vision projector (vision/mmproj)
    std::string arch;                   ///< the architecture name, for messages
};

/// The device bytes of `cells` cells of one attention layer's K and V in `format` (1 fp16, 2 int8 codes with a scale per
/// 64 values), the layout of a cell's row in both caches.
inline uint64_t kv_cell_bytes(const Geometry& g, int format) {
    return format == 2 ? (uint64_t) g.n_kv * (g.head_dim + g.head_dim / 64 * 2) * 2
                       : (uint64_t) g.n_kv * g.head_dim * 2 * 2;
}

/// Whether a stored drafter K/V is this session's: the same format and at least the matched rows, and not a ring (a ring
/// keeps cell `pos` at `pos % cells`, which no linear image a store can cut at a position describes).
inline bool mtp_image_matches(const Options& o, int format, int64_t cells, int64_t positions) {
    if (mtp_ring_cap(o.max_context, o.mtp_window) > 0) return false;
    return format == o.mtp_format && cells >= positions;
}

/// Why --mtp-kv/--mtp-window cannot run with these flags and the drafter the load picked (empty: they can).
inline std::string mtp_option_error(bool mtp_kv_given, const std::string& mtp_kv, int64_t mtp_window, Drafter drafter) {
    if (mtp_kv_given && mtp_kv_format(mtp_kv) < 0) return "--mtp-kv " + mtp_kv + ": fp16 (the default) or int8";
    if (mtp_window < 0 || mtp_window > ((int64_t) 1 << 40))
        return "--mtp-window " + std::to_string((long long) mtp_window) + ": 0 (every cell) or the cells the drafter reads";
    if ((mtp_kv_given || mtp_window > 0) && drafter != Drafter::Mtp)
        return std::string(mtp_kv_given ? "--mtp-kv" : "--mtp-window") + " needs the pack's MTP drafter (--spec 2..16)";
    return {};
}

class Model;
void destroy(Model* m);
struct ModelDeleter {
    void operator()(Model* m) const { destroy(m); }
};
using ModelPtr = std::unique_ptr<Model, ModelDeleter>;

/// The model's shared state: geometry, weights, and what its sessions use one at a time (scratch, the image encoder).
/// Sessions run one after another on the serving thread; each call leaves its stream idle. The decode windows of several
/// sessions (at most kWindow rows in all) run as one batch through `ModelSession::decode_batch`.
ModelPtr make_model(const Geometry& g, const Weights& w, const Options& o, std::string& err);
std::unique_ptr<model::ModelSession> make_session(Model& m, std::string& err);
/// The cells of each attention layer a session keeps on the card (kv_resident's rule).
int64_t resident_cells(int64_t max_context, int64_t kv_resident);
/// Device bytes one session takes: its KV cells on the card, recurrent states and window rows, the logits and the drafter's state.
uint64_t session_bytes(const Geometry& g, int64_t max_context, bool mtp, bool kv_int8 = false, int64_t kv_resident = 0,
                       int mtp_format = 1, int64_t mtp_window = 0);
/// Device bytes one session's cells and recurrent state take in layer `il`.
uint64_t session_layer_bytes(const Geometry& g, int il, int64_t max_context, bool kv_int8, int64_t kv_resident);
/// Page-locked host bytes one session's streamed K/V takes (0 without streaming).
uint64_t session_host_bytes(const Geometry& g, int64_t max_context, bool kv_int8, int64_t kv_resident);
/// Device bytes the model's shared scratch takes (with streaming, the VRAM copy a prompt chunk reads one layer's far cells from).
uint64_t shared_bytes(const Geometry& g, int64_t max_context, bool kv_int8 = false, int64_t kv_resident = 0);
/// Device bytes a session's block drafter keeps (its context ring and its window rows); 0 without one.
uint64_t block_session_bytes(const Geometry& g, const BlockDrafter& bd);

}  // namespace halo::dense
