// Load the control-vector files selected by the CLI.
#pragma once

#include "halo/models/qwen4exp/shared_names.hpp"
#include <string>

namespace halo::core {
struct ModelGeometry;
}

namespace halo::models::qwen4exp {
struct Options;
}

namespace halo::models::qwen4exp::cli {

bool load_control_vectors(const Options& o, const halo::core::ModelGeometry& g, std::string& summary,
                          std::string& error);

}  // namespace halo::models::qwen4exp::cli
