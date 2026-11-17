#pragma once

#include <optional>
#include <vector>
#include "halo/program/gpu_select.hpp"

namespace halo::program { struct CommonOptions; }

namespace halo::program::cli {

// An exit code for --gpu-plan or a failed choice; otherwise the one chosen card is applied and returned in `chosen_cards`.
std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards);

}  // namespace halo::program::cli
