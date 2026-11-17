// basalt/program/engine_local.hpp - program-level helpers and shapes used by more than one translation unit.
// Private to src/program: nothing here is part of the engine's public surface.
#pragma once

#include "basalt/models/qwen4exp/shared_names.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "basalt/core/expert_cache.hpp"
#include "basalt/models/qwen4exp/decode/expert_dispatch.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/models/qwen4exp/decode/native_dense.hpp"
#include "basalt/core/spin_wait.hpp"
#include "basalt/models/qwen4exp/decode/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"


namespace basalt::models::qwen4exp::engine_local {

// Spare RAM places for a saturated adaptation round.
inline constexpr int kSatBoost = 4;

/// The largest multiple of 256 in [0, ceiling] that `ok` accepts.  `ok` is monotone: false once too large.
template <class Ok>
int64_t biggest_chunk(int64_t ceiling, Ok&& ok) {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
}

// One unit of background work at a time, on a persistent worker thread (starting a thread per job costs more than
// the job's wake-up).  `join` waits for the job in flight.
class JobRunner {
public:
    JobRunner() = default;
    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;
    ~JobRunner() {
        join();
        if (th_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
            cv_.notify_one();
            th_.join();
        }
    }
    void run(std::function<void()> fn) {
        if (!th_.joinable()) th_ = std::thread([this] { loop(); });
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = std::move(fn);
            done_.store(false, std::memory_order_relaxed);
            has_ = true;
        }
        busy_ = true;
        cv_.notify_one();
    }
    bool joinable() const { return busy_; }
    bool finished() const { return !busy_ || done_.load(std::memory_order_acquire); }
    void join() {
        if (!busy_) return;
        // the caller has nothing else to do: poll rather than sleep, so the job's end is seen at once
        basalt::core::SpinWait spin;
        while (!done_.load(std::memory_order_acquire)) spin();
        busy_ = false;
    }
private:
    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [this] { return has_ || stop_; });
            if (!has_) return;
            std::function<void()> fn = std::move(fn_);
            has_ = false;
            lk.unlock();
            fn();
            done_.store(true, std::memory_order_release);
            lk.lock();
        }
    }
    bool busy_ = false;
    bool has_ = false, stop_ = false;
    std::atomic<bool> done_{true};
    std::mutex m_;
    std::condition_variable cv_;
    std::function<void()> fn_;
    std::thread th_;
};
/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    basalt::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// The routing trace: written once per layer from the calling thread, so it needs no lock.
    std::FILE* routing = nullptr;
};
struct SlotDelete {
    bool own = true;
    void operator()(basalt::core::SessionState* p) const { if (own) delete p; }
};
using SlotSession = std::unique_ptr<basalt::core::SessionState, SlotDelete>;

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out);
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer);

}  // namespace basalt::models::qwen4exp::engine_local
