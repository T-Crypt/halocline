// src/dense/profile.cpp - the CUDA side of --profile on the qwen35 path (WindowProfile): one session's decode windows
// stamped with timing events, summed per window width, printed through the engine log every 200 windows and at the
// session's end.
#include "profile_cuda.hpp"

#include "halo/log.hpp"

#include <algorithm>
#include <string>

namespace halo::dense {

namespace {
constexpr int kPrintEvery = 200;   // windows between the table's prints on a long serve
}  // namespace

WindowProfile::WindowProfile(int n_layers) {
    const size_t n = (size_t) std::max(n_layers, 1) * 6 + 320;
    bool ok = cudaEventCreate(&tail_) == cudaSuccess;
    for (Pool& p : pools_) {
        p.ev.assign(n, nullptr);
        p.kind.assign(n, Stage::None);
        for (cudaEvent_t& e : p.ev)
            if (cudaEventCreate(&e) != cudaSuccess) {
                (void) cudaGetLastError();
                e = nullptr;
                ok = false;
            }
    }
    if (ok) return;
    // not one mark is made: the caller keeps the session on its captured path
    for (Pool& p : pools_)
        for (cudaEvent_t& e : p.ev) {
            if (e != nullptr) cudaEventDestroy(e);
            e = nullptr;
        }
    if (tail_ != nullptr) cudaEventDestroy(tail_);
    tail_ = nullptr;
    halo::log::warn("halo dense profile", "the CUDA events for the window stamps could not be made; --profile is off");
}

WindowProfile::~WindowProfile() {
    for (Pool& p : pools_)
        for (cudaEvent_t& e : p.ev)
            if (e != nullptr) cudaEventDestroy(e);
    if (tail_ != nullptr) cudaEventDestroy(tail_);
    (void) cudaGetLastError();   // a context already torn down at process exit: nothing left to destroy
}

void WindowProfile::begin(int width, cudaStream_t s) {
    if (!ok()) return;
    Pool& next = pools_[1 - cur_];
    if (cudaEventRecord(next.ev[0], s) != cudaSuccess) {
        (void) cudaGetLastError();
        open_ = false;   // no start stamp: the window is not measured (a broken stream or context)
        return;
    }
    const bool was_open = open_;
    if (was_open) close(cur_, next.ev[0]);
    cur_ = 1 - cur_;
    Pool& p = pools_[cur_];
    p.n = 1;
    p.kind[0] = Stage::None;
    width_ = width;
    open_ = true;
    dropped_ = false;
}

void WindowProfile::mark(Stage st, cudaStream_t s) {
    if (!ok() || !open_) return;
    Pool& p = pools_[cur_];
    if ((size_t) p.n >= p.ev.size()) {
        if (!warned_) {
            halo::log::warn("halo dense profile", "a window makes more marks than the pool holds; its stages are skipped");
            warned_ = true;
        }
        dropped_ = true;
        return;
    }
    if (cudaEventRecord(p.ev[(size_t) p.n], s) != cudaSuccess) {
        (void) cudaGetLastError();
        dropped_ = true;
        return;
    }
    p.kind[(size_t) p.n] = st;
    ++p.n;
}

void WindowProfile::finish(cudaStream_t s) {
    if (!ok()) return;
    if (open_) {
        if (cudaEventRecord(tail_, s) == cudaSuccess) close(cur_, tail_);
        else (void) cudaGetLastError();
        open_ = false;
    }
    if (table_.windows() > 0) print();
}

void WindowProfile::close(int pool, cudaEvent_t end) {
    Pool& p = pools_[pool];
    if (p.n < 1) return;
    if (dropped_) {
        ++skipped_;
    } else {
        // every stamp is on the stream before `end`, so the host is about to wait for it anyway (the caller's window
        // ends in a stream sync): the wait costs nothing it does not already pay and no event below is still pending
        if (cudaEventSynchronize(end) != cudaSuccess) (void) cudaGetLastError();
        double stage_ms[kStageCount] = {};
        float ms = 0.0f;
        for (int k = 1; k < p.n; ++k) {
            if (p.kind[(size_t) k] == Stage::None) continue;   // a boundary's interval is the gap
            if (cudaEventElapsedTime(&ms, p.ev[(size_t) k - 1], p.ev[(size_t) k]) != cudaSuccess) {
                (void) cudaGetLastError();
                continue;
            }
            stage_ms[(int) p.kind[(size_t) k]] += (double) ms;
        }
        if (cudaEventElapsedTime(&ms, p.ev[0], end) != cudaSuccess) (void) cudaGetLastError();
        table_.add(width_, (double) ms, stage_ms);
        ++since_print_;
    }
    p.n = 0;
    dropped_ = false;
    open_ = false;
    if (since_print_ >= kPrintEvery) print();
}

void WindowProfile::print() {
    for (const std::string& line : table_.report(skipped_)) halo::log::info("halo dense profile", "%s", line.c_str());
    table_.clear();
    since_print_ = 0;
    skipped_ = 0;
}

}  // namespace halo::dense
