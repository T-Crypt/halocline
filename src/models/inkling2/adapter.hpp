// src/models/inkling2/adapter.hpp - the inkling2 model's registration entry (src/models/builtin.cpp).
#pragma once
#include "halo/model/adapter.hpp"

#include <memory>

namespace halo::models::inkling2 {
std::unique_ptr<model::ModelAdapter> make_adapter();
}
