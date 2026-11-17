// src/models/qwen35/adapter.hpp - the qwen35 (Qwen3.5-family dense hybrid) adapter.
#pragma once

#include "halo/model/adapter.hpp"

#include <memory>

namespace halo::models::qwen35 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}  // namespace halo::models::qwen35
