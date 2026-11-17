#pragma once

#include "halo/model/adapter.hpp"
#include "halo/program/options.hpp"

namespace halo::program::generate {

/// A model served through its sessions, to its exit code.
int run_session_model(halo::program::CommonOptions& o, const halo::model::ModelAdapter& adapter);

}  // namespace halo::program::generate
