#pragma once

#include "halo/models/qwen4exp/decode/mtp.hpp"
#include "halo/models/qwen4exp/decode/conversation_snapshot.hpp"
#include "halo/models/qwen4exp/decode/verify.hpp"
#include "halo/core/diag.hpp"
#include "halo/core/draft_cum.hpp"
#include "halo/models/qwen4exp/decode/kv_arena.hpp"
#include "halo/core/direct_upload.hpp"
#include "halo/core/load_progress.hpp"
#include "halo/core/on_device.hpp"

#include "halo/models/qwen4exp/decode/native_head.hpp"
#include "halo/platform/file.hpp"
#include "halo/platform/memory.hpp"
#include "halo/kernels/bf16_gemv.hpp"
#include "halo/kernels/cpu/expert.hpp"
#include "halo/models/qwen4exp/geometry.hpp"
#include "halo/kernels/elementwise.hpp"
#include "halo/models/qwen4exp/kernels/fused_gr.hpp"
#include "halo/models/qwen4exp/kernels/gr.hpp"
#include "halo/models/qwen4exp/kernels/hc_fused.hpp"
#include "halo/models/qwen4exp/kernels/kv_q4.hpp"
#include "halo/models/qwen4exp/kernels/kv_q8.hpp"
#include "halo/models/qwen4exp/kernels/native_moe.hpp"
#include "halo/kernels/native_mmvq.hpp"
#include "halo/models/qwen4exp/kernels/native_qsa.hpp"
#include "halo/models/qwen4exp/kernels/native_rope.hpp"
#include "halo/models/qwen4exp/kernels/native_router.hpp"
#include "halo/models/qwen4exp/kernels/qsa.hpp"
#include "halo/models/qwen4exp/kernels/qsa_decode_attn.hpp"
#include "halo/models/qwen4exp/kernels/qsa_dec_attn.hpp"
#include "halo/models/qwen4exp/kernels/qsa_dec_fused.hpp"
#include "halo/models/qwen4exp/kernels/q5k_head.hpp"
#include "halo/models/qwen4exp/kernels/q6k_head.hpp"
#include "halo/models/qwen4exp/kernels/q80_proxy.hpp"
#include "halo/kernels/quantize_act.hpp"
#include "halo/models/qwen4exp/kernels/rope.hpp"
#include "halo/models/qwen4exp/kernels/router_top10.hpp"
#include "halo/models/qwen4exp/kernels/s2_expert_grouped.hpp"
#include "halo/kernels/sampler.hpp"
#include "halo/models/qwen4exp/kernels/shared_expert.hpp"
#include "halo/kernels/head_rows.hpp"
#include "halo/models/qwen4exp/kernels/verify_kernels.hpp"
#include "halo/models/qwen4exp/kernels/ngram.hpp"
#include "halo/models/qwen4exp/kernels/dense_tc.hpp"
#include "halo/kernels/pdl.hpp"

#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>
#include "halo/platform/compiler.hpp"
#include "halo/log.hpp"


namespace halo::core::mtp_detail {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;

// spare rows the draft head carries after the base rows, for the head rows of tokens the subset lacks
constexpr int64_t kDraftCtxRows = 512;
// of them, the rows a prompt's tokens may take: the rest stay free for the reply's own
constexpr int64_t kDraftCtxPromptRows = 384;
// the rows the target's candidates leave free for the reply's own picks
constexpr int64_t kDraftCtxPickRows = 64;

double ms_since(Clock::time_point t);
halo::kernels::QsaShapes shapes_of(const ModelGeometry& g);
bool read_file(const FileRange& r, std::vector<uint8_t>& out);

}
