#include "common.hpp"
#include "expert_plan.hpp"
#include "halo/models/qwen4exp/decode/verify.hpp"
#include "halo/core/on_device.hpp"
#include "halo/platform/compiler.hpp"
#include "halo/kernels/cpu/expert_layout.hpp"
#include "halo/models/qwen4exp/geometry.hpp"
#include "halo/kernels/pdl.hpp"
#include "halo/models/qwen4exp/kernels/dec_moe.hpp"
#include <atomic>
#include <string>

namespace halo::core {
using namespace verify_detail;

// The device-planned tail's window start: residency tables and the flag words.
bool Verifier::dm_begin(std::string& err, DecodeGaps* gm) {
    if (!dm_sync(err)) return false;
    decode_events().stamp(4);
    if (gm) gm->mark(13);
    *(volatile uint32_t*) dm_->h_stop = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (gm) gm->mark(14);
    return true;
}

}  // namespace halo::core
