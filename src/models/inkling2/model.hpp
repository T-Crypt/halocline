// src/models/inkling2/model.hpp - Inkling 2 Mini (Thinking Machines, `inkling2`): a looped MoE transformer of 24 stored
// layers and 48 layer visits (prelude 0-5 once, core 6-17 three passes with shared weights, coda 18-23 once), block
// attention residuals, relative-bias attention with a gate and a sink (20 sliding-window layers of 1,024, 4 full),
// short convolutions, and 128 routed experts (top 8) in NVFP4 (layers 0-19) and Q8_0 (20-23, MXFP8 in the release).
// Every visit keeps its own K/V and convolution state. The math follows llama.cpp's inkling2 graph (Unsloth
// b12588-mix-c0c3de8, src/models/inkling2.cpp), the numerics oracle of the branch halo-inkling.
#pragma once
#include "halo/model/adapter.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace halo {
class GgufFile;
}

namespace halo::models::inkling2 {

constexpr int kTypeF32 = 0, kTypeQ8_0 = 8, kTypeBf16 = 30, kTypeNvfp4 = 40;

/// The shape the GGUF's inkling2.* keys give.
struct Hparams {
    int n_embd = 0, n_head = 0, n_kv = 0, head_dim = 0;
    int n_phys = 0;          ///< stored layers (24)
    int n_visit = 0;         ///< layer visits (48)
    int core_start = 0, core_end = 0, n_pass = 1, gate_rank = 0;
    int n_expert = 0, n_used = 0, n_ff = 0;
    int64_t n_vocab_padded = 0, n_vocab = 0, context_length = 0;
    int swa = 0, rel_len = 0, rel_extent = 0, conv = 0, res_block = 0;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_floor = 0, log_alpha = 0;
    float situ_beta = 0, situ_lbeta = 0, w_scale = 1, softcap = 0;
    std::vector<uint8_t> is_swa;   ///< per visit
    /// The stored layer a visit runs.
    int phys(int v) const {
        const int n_core = core_end - core_start;
        if (n_pass <= 1 || v < core_start) return v;
        if (v < core_start + n_core * n_pass) return core_start + (v - core_start) % n_core;
        return v - n_core * (n_pass - 1);
    }
    /// The pass of a core visit (0 outside the core).
    int pass(int v) const {
        const int n_core = core_end - core_start;
        if (n_pass <= 1 || v < core_start || v >= core_start + n_core * n_pass) return 0;
        return (v - core_start) / n_core;
    }
    bool in_core(int v) const {
        const int n_core = core_end - core_start;
        return n_pass > 1 && v >= core_start && v < core_start + n_core * n_pass;
    }
};

/// Reads and checks the hparams; an error names what is missing or out of range.
bool read_hparams(const GgufFile& gg, Hparams& h, std::string& err);

/// The loaded model (weights on the card) and its sessions.
std::unique_ptr<model::LoadedModel> load_model(const model::LoadRequest& req, std::string& err);

}  // namespace halo::models::inkling2
