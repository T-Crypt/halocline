#include "state.hpp"
#include "layout.hpp"
#include "halo/models/qwen4exp/prefill/prefill.hpp"
#include "halo/models/qwen4exp/decode/mtp.hpp"
#include "halo/core/host_plan.hpp"
#include <string>

namespace halo::prefill {
using namespace detail;

bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
    const bool body_ok = run_impl(tokens, n, pos0, err);
    // the tokens the warm-up reads belong to this request: wait for it once the run that reads them is over
    impl_->warm.pending = false;
    if (impl_->warm.fut.valid() && (!body_ok || pos0 + n >= impl_->warm.to)) impl_->warm.fut.wait();

    return body_ok;
}

}  // namespace halo::prefill
