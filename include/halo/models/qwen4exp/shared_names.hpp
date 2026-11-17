// include/halo/models/qwen4exp/shared_names.hpp - this model's namespaces see the shared server's and tools' names.
#pragma once

namespace halo::program {
namespace cli {}
namespace serve {}
namespace tokenizer {}
}  // namespace halo::program

namespace halo::models::qwen4exp {
using namespace halo::program;
namespace cli { using namespace halo::program::cli; }
namespace serve { using namespace halo::program::serve; }
namespace tokenizer { using namespace halo::program::tokenizer; }
}  // namespace halo::models::qwen4exp
