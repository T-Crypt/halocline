// src/models/qwen4exp/serve/qfn_runtime.hpp - the persistent driver over startup's loaded engine.
#pragma once

#include "halo/models/qwen4exp/shared_names.hpp"
#include "../../../program/serve/state.hpp"
#include "../load/engine_local.hpp"
#include "halo/program/expert_spill.hpp"
#include "halo/program/grammar/guard.hpp"
#include "halo/platform/system.hpp"

#include <functional>
#include <optional>

namespace halo::core {
class ModelFile;
class MtpDrafter;
}  // namespace halo::core
namespace halo::kernels { class PleTable; }
namespace halo::kernels::cpu { class ExpertPool; }
namespace halo::vision { class CudaEncoder; }

namespace halo::models::qwen4exp {
struct Options;
namespace serve {

// All engine objects belong to startup and outlive run(). Callbacks borrow the same objects.
struct Context {
    Options& o;
    std::string& ppl_base;
    std::string& ppl_out;
    halo::core::ModelFile& model_file;
    const bool& no_spec;
    std::string& err;
    halo::core::ModelGeometry& g;
    int64_t& K;
    std::vector<std::pair<int32_t, int32_t>>& profile;
    std::vector<std::pair<int32_t, int32_t>>& profile_loaded;
    const bool& pf_borrow;
    halo::core::WeightTable& wt;
    int32_t*& d_mrope;
    std::vector<int32_t>& mrope_host;
    std::string& cvec_summary;
    halo::core::SessionState& ss;
    cudaStream_t& main_stream;
    void* const& main_cs;
    halo::kernels::PleTable& ple_table;
    halo::vision::CudaEncoder& vis_enc;
    const int64_t& kv_floor;
    std::vector<std::vector<engine_local::SlotSession>>& bslot_ss;
    std::vector<int64_t>& slot_cap;
    bool& slot0_is_solo;
    halo::core::MtpDrafter& mtp;
    const bool& mtp_on;
    const halo::core::QsaState* const& mtp_kv;
    const int64_t& n_vocab;
    std::optional<halo::program::grammar::PackVocab>& grammar_vocab;
    halo::core::NativeHead& native_head;
    halo::core::FileExpertSource& src;
    halo::core::ExpertSource*& srcp;
    halo::core::ExpertCache& xcache;
    engine_local::Drive& drive;
    std::vector<int32_t>& host_res;
    std::function<cudaError_t(int32_t*)> res_put;
    int32_t*& d_res;
    halo::core::TokenHits& thits;
    const bool& graph_hits;
    int64_t& auto_lend_pct;
    int64_t& auto_ceiling;
    halo::kernels::cpu::ExpertPool& pool;
    std::function<int64_t()> read_cap;
    halo::program::ExpertSpill& tier_spill;
    std::function<void()> report_expert_tiers;
};

// Serves until QUIT or stdin closes; returns the process exit code.
int run(Context& context);

}  // namespace serve
}  // namespace halo::models::qwen4exp
