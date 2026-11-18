// src/dense/profile_cuda.hpp - --profile's CUDA events on one session's decode windows.
//
// A window is stamped at its stage boundaries on the window's stream; the interval that ends at a mark is added to the
// stage the mark names, and the window's wall runs from its first mark to the next window's first mark (so the drafter
// work that follows a window and the host work between them are in the same wall). A boundary mark (`Stage::None`)
// ends an interval that belongs to no stage, which keeps a stretch of host work out of the next stage's ms. Nothing is
// recorded inside a CUDA graph: a session with a profile never captures its window (src/dense/forward.cpp).
#pragma once

#include "halo/dense/profile.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace halo::dense {

/// One session's window table. The metrics are deliberately not for A/B: its marks are host-side event records, the
/// window runs as launched (no graph replay), and the window body runs without PDL, so overlaps the plain path hides
/// behind a graph are visible here.
class WindowProfile {
public:
    static constexpr bool kOn = true;   // window_body's stages type: PDL off, every mark recorded

    /// `n_layers` sizes the event pool: six marks per layer (four of an attention layer's and two of its FFN's, three
    /// of a GDN layer's and two of its FFN's) plus every mark outside the layer loop and a wide margin.
    explicit WindowProfile(int n_layers);
    ~WindowProfile();
    WindowProfile(const WindowProfile&) = delete;
    WindowProfile& operator=(const WindowProfile&) = delete;

    /// False when no event could be made: the caller then leaves the session exactly as without --profile.
    bool ok() const { return !pools_[0].ev.empty(); }

    /// Opens a window of `width` rows: records its first stamp, and closes the previous one against that same stamp
    /// (the previous window's wall ends where this one starts).
    void begin(int width, cudaStream_t s);
    /// Records a stage boundary, adding the interval since the previous stamp to `st`.
    void mark(Stage st, cudaStream_t s);
    /// Closes the open window (its wall ends at this stamp), prints what the table holds, and frees nothing.
    void finish(cudaStream_t s);

private:
    struct Pool {
        std::vector<cudaEvent_t> ev;
        std::vector<Stage> kind;
        int n = 0;   // stamps recorded: the first is the window's start
    };
    void close(int pool, cudaEvent_t end);
    void print();

    ProfileTable table_;
    Pool pools_[2];   // the window in hand and the next one's first stamp, so closing never reuses a live event
    cudaEvent_t tail_ = nullptr;
    int cur_ = 0, width_ = 0;
    bool open_ = false, dropped_ = false, warned_ = false;
    long skipped_ = 0, since_print_ = 0;
};

}  // namespace halo::dense
