#pragma once

#include <optional>
#include <string>

namespace halo::program { struct CommonOptions; }
namespace halo::model { class ModelAdapter; }

namespace halo::program::cli {

// An exit code for help or an invalid argument; otherwise startup can continue.
std::optional<int> parse_options(int argc, char** argv, CommonOptions& o, const model::ModelAdapter* adapter);

}  // namespace halo::program::cli
