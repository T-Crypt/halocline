// src/program/serve/session_engine.hpp - serving a model that offers only a ModelSession.
#pragma once

#include "request.hpp"
#include "halo/model/adapter.hpp"

namespace halo::program::serve {

/// Serves until QUIT or stdin closes; returns the process exit code.
int run_session(CommonOptions& o, halo::core::ModelFile& file, const halo::model::ModelAdapter& adapter,
                halo::model::LoadedModel& model);

}  // namespace halo::program::serve
