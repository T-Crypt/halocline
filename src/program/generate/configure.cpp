#include "run.hpp"
#include "session_model.hpp"
#include "../cli/gpu.hpp"
#include "../cli/startup.hpp"

#include "halo/model/adapter.hpp"
#include "halo/models/builtin.hpp"
#include "halo/program/gpu_select.hpp"
#include "halo/program/options.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace halo::program::generate {

namespace cli = halo::program::cli;

int run(int argc, char** argv) {
    halo::models::register_builtin_models();
    std::unique_ptr<halo::program::CommonOptions> options;
    const halo::model::ModelAdapter* adapter = nullptr;
    if (const auto code = cli::start(argc, argv, options, adapter)) return *code;
    halo::program::CommonOptions& common = *options;
    // ---- the GPUs: chosen before any CUDA call, which then sees only them, the main card as device 0
    std::vector<halo::program::GpuCard> chosen_cards;
    if (const auto code = cli::select_gpus(common, chosen_cards)) return *code;
    // ---- where every loader reads from: the sections of the .halo file
    if (const auto code = cli::prepare_model_process(common)) return *code;
    if (const std::optional<int> code = adapter->run(common, chosen_cards)) return *code;
    return run_session_model(common, *adapter);
}

}
