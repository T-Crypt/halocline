#include "halo/models/builtin.hpp"

#include "halo/model/adapter.hpp"

#include <memory>

// One HALO_BUILT_MODEL(<arch>) line per model in HALO_MODELS, written by cmake/models.cmake. Each model's adapter
// factory is halo::models::<arch>::make_adapter(), declared here so the model's headers stay out of this file.
namespace halo::models {
#define HALO_BUILT_MODEL(arch) \
    namespace arch { std::unique_ptr<model::ModelAdapter> make_adapter(); }
#include "builtin_models.inc"
#undef HALO_BUILT_MODEL

void register_builtin_models() {
#define HALO_BUILT_MODEL(arch) \
    if (model::find_adapter(#arch) == nullptr) model::register_adapter(arch::make_adapter());
#include "builtin_models.inc"
#undef HALO_BUILT_MODEL
}

}  // namespace halo::models
