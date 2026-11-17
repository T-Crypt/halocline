#include "common.hpp"
#include "expert_plan.hpp"
#include "diagnostics.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/log.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <immintrin.h>

namespace basalt::core {
using namespace verify_detail;

namespace verify_detail {
std::atomic<Verifier*> g_live[kLiveMax];
std::atomic<const Verifier*> g_diag_verifier{nullptr};
void diag_active_verifier(std::FILE* f) {
    int live = 0;
    for (auto& slot : g_live) live += slot.load() != nullptr ? 1 : 0;
    if (live > 1) {   // a layer split's stages (and a pipelined pair's chains): each one's window
        for (auto& slot : g_live)
            if (const Verifier* v = slot.load()) v->diag(f);
        return;
    }
    if (const Verifier* v = g_diag_verifier.load()) v->diag(f);
}
// every live verifier (a layer split has one per stage), for the release before the engine ends
void release_live_verifiers(std::FILE* f) {
    for (auto& slot : g_live)
        if (Verifier* v = slot.load()) {
            const Clock::time_point t0 = Clock::now();
            const bool done = v->release_gpu_waits(5000);
            if (f != nullptr)
                std::fprintf(f, "basalt: released the verify window's GPU waits: the GPU %s\n",
                             done ? ("finished in " + std::to_string((long long) ms_since(t0)) + " ms").c_str()
                                  : "did not finish within 5 s");
        }
    if (f != nullptr) std::fflush(f);
}
std::string released_note(bool drained) {
    return drained ? "; its GPU waits were released and the GPU finished"
                   : "; its GPU waits were released but the GPU did not finish within 5 s";
}
}

bool Verifier::release_gpu_waits(int timeout_ms) {
    released_.store(true);
    // the words the spin kernels read (wait_flag_ge, wait_flag_ge_or) are mapped host memory, so a store here reaches
    // them with no API call; UINT32_MAX is past every ring.  (The skip words are device memory, but wait_flag_ge_or
    // also returns on its flag.)  A host function raising flag B later only raises.
    for (uint32_t* p : {h_flag_, h_flagA_, h_flagB_})
        if (p != nullptr) *(volatile uint32_t*) p = UINT32_MAX;
    if (h_flag_ != nullptr) {
        *(volatile uint32_t*) (h_flag_ + 8) = UINT32_MAX;
        *(volatile uint32_t*) (h_flag_ + 12) = UINT32_MAX;   // the hand-off a later stage's graph waits for
    }
    basalt::kernels::wait_abort_release();   // the bounded device waits (mid chain, commit finisher) fall through
    if (dm_ != nullptr) *(volatile uint32_t*) dm_->h_stop = UINT32_MAX;   // the glue launches' landed wait ends
    std::atomic_thread_fence(std::memory_order_seq_cst);
    _mm_sfence();
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    for (cudaStream_t s : {cs_}) {
        if (s == nullptr) continue;
        while (cudaStreamQuery(s) == cudaErrorNotReady) {
            if (ms_since(t0) > timeout_ms) {
                gpu_left_running().store(true);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    return true;
}

std::string Verifier::dm_note() const {
    const DecMoe* d = dm_;
    if (d == nullptr) return "";
    auto rd = [](const uint32_t* p) { return *(const volatile uint32_t*) p; };
    const uint32_t seq = rd(h_seq_);
    std::string s = "; device plan: ring " + std::to_string(seq) + " published";
    if (seq >= 1 && (int64_t) seq <= d->rings) {
        const basalt::kernels::DmRing& r = d->h_ring[seq - 1];
        s += " (entries: VRAM " + std::to_string(r.n_vram) + ", CPU " +
             std::to_string(r.n_cpu) + ")";
    }
    s += ", host flag " + std::to_string(rd(h_flag_));
    return s;
}

void Verifier::diag(std::FILE* f) const {
    auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
    // outside a verify stage these are the last window's numbers (it finished), not the stalled work's
    const char* where = progress().where.load();
    const bool current = where != nullptr && std::strncmp(where, "verify window", 13) == 0;
    std::fprintf(f, "  verify window%s (layers %lld-%lld, device %d): %d tokens at position %lld, host at layer step %u; the GPU rang %u; flags: "
                    "served %u, plan (A) %u, copies (B) %u, hand-off %u\n", current ? "" : " (last window, not the current stage)",
                 (long long) lb_, (long long) (le_ - 1), device_, last_t_, (long long) last_pos0_, cur_layer_ + 1, rd(h_seq_), rd(h_flag_),
                 rd(h_flagA_), rd(h_flagB_), rd(h_flag_ + 12));
    if (dm_ != nullptr) std::fprintf(f, "  %s\n", dm_note().c_str() + 2);
}

}  // namespace basalt::core
