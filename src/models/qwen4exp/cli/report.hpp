#pragma once

#include "halo/models/qwen4exp/shared_names.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace halo::models::qwen4exp { struct Options; }
namespace halo::models::qwen4exp::engine_local { struct Drive; }
namespace halo::core {
struct ModelGeometry;
struct TokenGraph;
struct SessionGraphs;
class ExpertSource;
class FileExpertSource;
class ExpertCache;
class Verifier;
class MtpDrafter;
}
namespace halo::kernels {
class PleTable;
namespace cpu { class ExpertPool; }
}

namespace halo::models::qwen4exp::cli {

struct StatsReport {
    const Options& o;
    const halo::core::ModelGeometry& g;
    const engine_local::Drive& drive;
    const halo::kernels::cpu::ExpertPool& pool;
    const halo::kernels::PleTable& ple_table;
    const halo::core::ExpertSource* srcp;
    const halo::core::ExpertCache& xcache;
    const halo::core::TokenGraph& tgraph;
    const halo::core::SessionGraphs& gr;
    double decode_ms;
    int64_t phase_tokens;
    double ms_ple;
    double ms_embed;
    double ms_layers;
    double ms_head;
    double ms_readback;
    double ms_sample;
};

struct SpeculationReport {
    const Options& o;
    const halo::core::ModelGeometry& g;
    const engine_local::Drive& drive;
    const halo::kernels::cpu::ExpertPool& pool;
    const halo::core::FileExpertSource& src;
    const halo::core::ExpertSource* srcp;
    const halo::core::Verifier& ver;
    const halo::core::MtpDrafter& mtp;
    int64_t rounds;
    int64_t drafts_ok;
    int64_t drafts_total;
    const std::vector<int64_t>& window_hist;
    const std::vector<int64_t>& accepted_hist;
    int64_t sfx_windows;
    int64_t sfx_ok;
    int64_t sfx_drafts;
    int64_t misses0;
    int64_t entries0;
    int64_t swaps_total;
    int64_t ram0;
    int64_t files0;
    int64_t pcie0;
    uint64_t fall0;
    double pool_ms0;
    double ms_adapt;
    bool use_mtp;
};

void print_stats(const StatsReport& report);
void print_speculation(const SpeculationReport& report);
void print_routing_dump(const std::string& path, int64_t records);
double print_generation(const Options& o, const std::vector<int64_t>& produced, int64_t n_prompt,
                        double total_ms, double prefill_ms, double ttft_ms);

}  // namespace halo::models::qwen4exp::cli
