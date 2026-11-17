#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

// A host-to-device upload in pieces of at most kSmallCopy bytes: a copy that small passes the expert copies queued on
// the card's copy engine (an adaptive round's), a larger one waits behind all of them.
constexpr size_t kSmallCopy = 16384;
static cudaError_t copy_small(void* dst, const void* src, size_t bytes, cudaStream_t s) {
    for (size_t o = 0; o < bytes; o += kSmallCopy) {
        const cudaError_t e = cudaMemcpyAsync((uint8_t*) dst + o, (const uint8_t*) src + o, std::min(kSmallCopy, bytes - o),
                                              cudaMemcpyHostToDevice, s);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// The residency table the window classifies with: host_res as it is now (between windows).
// The upload is queued on the stream that reads it (CUDA0's window): a plain cudaMemcpy from pageable
// memory may return before its DMA lands, and the window's route kernel would then read a table half old, half new
// (an expert with two tiers in one ring).
bool Verifier::dm_sync(std::string& err) {
    DecMoe* d = dm_;
    const size_t LE = d->shadow0.size();
    auto span = [&](const int32_t* now, std::vector<int32_t>& sh, size_t& a, size_t& b) {
        a = LE; b = 0;
        if (std::memcmp(now, sh.data(), LE * 4) == 0) return false;
        for (size_t i = 0; i < LE; ++i)
            if (now[i] != sh[i]) { if (a == LE) a = i; b = i + 1; }
        if (a < b) std::memcpy(sh.data() + a, now + a, (b - a) * 4);
        return a < b;
    };
    size_t a = 0, b = 0;
    uint8_t* pin = nullptr;
    if (d->pin != nullptr) {
        pin = d->pin + (size_t) d->pin_i * d->pin_stride;
        d->pin_i = (d->pin_i + 1) % DecMoe::kPinSlots;
    }
    int32_t* const pin_res0 = (int32_t*) pin;
    unsigned long long* const pin_ptab0 = (unsigned long long*) (pin + (pin ? (size_t) LE * 4 : 0));
    const bool up0 = span(hits_.h_res, d->shadow0, a, b);
    if (up0) {
        decode_events().tag(1u << 4);
        const int32_t* src = d->shadow0.data() + a;
        if (pin) { std::memcpy(pin_res0 + a, src, (b - a) * 4); src = pin_res0 + a; }
        if (copy_small(d->res0 + a, src, (b - a) * 4, cs_) != cudaSuccess) {
            err = "verify: device-planned MoE residency upload";
            return false;
        }
    }
    if (d->ptab0 != nullptr && a < b) {   // the same span as blob addresses (a < b only after an upload above)
        std::vector<unsigned long long> pv;
        unsigned long long* p = pin_ptab0 + a;
        if (!pin) { pv.resize(b - a); p = pv.data(); }
        for (size_t i = a; i < b; ++i) {
            const int32_t slot = d->shadow0[i];
            p[i - a] = slot >= 0 ? (unsigned long long) (hits_.cache_base + (hits_.slot_off ? (size_t) hits_.slot_off[slot]
                                                                                             : (size_t) slot * hits_.blob))
                                 : 0ull;
        }
        if (copy_small(d->ptab0 + a, p, (b - a) * 8, cs_) != cudaSuccess) {
            err = "verify: glue table upload";
            return false;
        }
    }
    return true;
}

unsigned int* Verifier::landed_words() const { return dm_ != nullptr && dm_->landed_pin != nullptr ? dm_->landed : nullptr; }

bool Verifier::landed_ok() const {
    if (landed_words() == nullptr || !landed_glue_) return false;
    for (int t = 1; t <= max_t_; ++t)
        if ((ar_off_ ? exec_nr_[t] : exec_[t]) == nullptr) return false;
    return true;
}

bool Verifier::landed_need(uint32_t round, std::string& err) {
    if (landed_words() == nullptr) return true;
    const OnDevice on_device(device_);
    uint32_t* src = dm_->landed_pin + dm_->landed_i;
    dm_->landed_i = (dm_->landed_i + 1) % DecMoe::kPinSlots;
    *src = round;
    if (cudaMemcpyAsync(dm_->landed + dm_->rings, src, 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "verify: the landed words' upload";
        return false;
    }
    return true;
}

bool Verifier::dm_fence(cudaStream_t s) {
    if (dm_ == nullptr) return true;
    const OnDevice on_device(device_);
    if (fence_ev_ == nullptr && cudaEventCreateWithFlags(&fence_ev_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        fence_ev_ = nullptr;
        return false;
    }
    return cudaEventRecord(fence_ev_, cs_) == cudaSuccess && cudaStreamWaitEvent(s, fence_ev_, 0) == cudaSuccess;
}

}  // namespace basalt::core
