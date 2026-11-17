#pragma once

#include <optional>
#include <string>

#include <memory>

namespace halo::program { struct CommonOptions; }
namespace halo::model { class ModelAdapter; }

namespace halo::program::cli {

// Finds the model named by --model and parses the flags into its options object.
std::optional<int> start(int argc, char** argv, std::unique_ptr<CommonOptions>& o, const model::ModelAdapter*& adapter);
std::optional<int> prepare_model_process(const CommonOptions& o);

}  // namespace halo::program::cli
