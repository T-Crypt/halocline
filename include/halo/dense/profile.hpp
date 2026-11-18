// include/halo/dense/profile.hpp - the dense decode window's stage table (--profile): per window width, the windows
// seen, their mean ms, and each stage's mean ms and share of it. Host only; the stamps themselves are in
// src/dense/profile_cuda.hpp, so the table and its text are tested without a card.
#pragma once

#include <array>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace halo::dense {

/// The widest decode window the table covers (kWindow, src/dense/internal.hpp asserts the two agree).
inline constexpr int kProfileWidths = 16;

/// The stages a window is stamped at. `None` is a boundary: the interval that ends there belongs to no stage (it is
/// the gap), so a mark before a stretch of host work keeps that work out of the next stage's ms.
enum class Stage : int {
    Embed = 0,     ///< the window's token embedding into the BF16 residual
    AttnPre,       ///< an attention layer's norm and its q/k/v product
    AttnQk,        ///< attn_qk_prepare (the cache write, rope and the norms)
    AttnCore,      ///< gqa_decode_attn
    AttnOut,       ///< the gate and the output projection (the 4-bit cache's rotation too)
    GdnPre,        ///< a GDN layer's norm and its qkv/z products
    GdnRec,        ///< dec_gdn_rec
    GdnOut,        ///< the GDN output projection
    FfnUp,         ///< the FFN's norm and its gate/up product
    FfnDown,       ///< the down product
    Head,          ///< the final norm and the LM head
    Sample,        ///< sample_tokens (accept is host-side, so it lands in gap)
    MtpDraft,      ///< one mtp_window call of the MTP drafter's draft chain
    MtpHist,       ///< one mtp_window call of the drafter's history (settle)
    None,          ///< a boundary, not a column
    Count
};

/// The number of columns: the stages up to `None`.
inline constexpr int kStageCount = (int) Stage::None;

/// The stage's column name ("attn-pre", "mtp-hist", ...), "-" for `None`.
inline const char* stage_name(Stage s) {
    static const char* const names[kStageCount + 1] = {"embed",    "attn-pre", "attn-qk",  "attn-core", "attn-out",
                                                       "gdn-pre",  "gdn-rec",  "gdn-out",  "ffn-up",    "ffn-down",
                                                       "head",     "sample",   "mtp-draft", "mtp-hist", "-"};
    const int i = (int) s;
    return i >= 0 && i <= kStageCount ? names[i] : "-";
}

/// Windows and their stage ms, summed, per row count T.
struct ProfileTable {
    struct Row {
        long windows = 0;
        double wall_ms = 0;
        double stage_ms[kStageCount] = {};
    };

    /// Adds one closed window. A width below 1 or above kProfileWidths is not part of a decode window's range and is
    /// dropped; stage times below zero (a clock that ran backwards) count as zero.
    void add(int width, double wall_ms, const double* stage_ms) {
        if (width < 1 || width > kProfileWidths || !(wall_ms >= 0.0)) return;
        Row& r = rows_[(size_t) width];
        ++r.windows;
        r.wall_ms += wall_ms;
        for (int i = 0; i < kStageCount; ++i) r.stage_ms[i] += stage_ms[i] > 0.0 ? stage_ms[i] : 0.0;
    }

    long windows() const {
        long n = 0;
        for (const Row& r : rows_) n += r.windows;
        return n;
    }

    void clear() { rows_ = {}; }

    /// The report's lines: a header (what the numbers are, and that the window ran uncaptured so a stage's ms holds
    /// the launch gap before its first kernel), then one line per width with windows. Each stage line carries its
    /// mean ms and its share of the mean window; the line closes with the sum of the stages against the wall, whose
    /// difference is the gap. Empty when no window was added. `skipped` counts windows whose marks overflowed the
    /// event pool and were dropped.
    std::vector<std::string> report(long skipped = 0) const {
        std::vector<std::string> out;
        const long total = windows();
        if (total == 0) return out;
        char b[96];
        std::string head;
        std::snprintf(b, sizeof b, "%ld decode windows", total);
        head = b;
        if (skipped > 0) {
            std::snprintf(b, sizeof b, ", %ld skipped (mark pool full)", skipped);
            head += b;
        }
        head += ": mean ms per window by its row count T; GPU events on the window's stream, the window uncaptured "
                "(PDL off) and its wall from its first stamp to the next window's first, so a stage's ms holds the "
                "launch gap before its own first kernel; accept is host-side and lands in gap";
        out.push_back(std::move(head));
        for (int t = 1; t <= kProfileWidths; ++t) {
            const Row& r = rows_[(size_t) t];
            if (r.windows == 0) continue;
            const double wm = r.wall_ms / (double) r.windows;
            std::snprintf(b, sizeof b, "T %d windows %ld window %.3f ms", t, r.windows, wm);
            std::string line = b;
            double sum = 0;
            for (int i = 0; i < kStageCount; ++i) {
                const double ms = r.stage_ms[i] / (double) r.windows;
                if (ms <= 0.0) continue;
                sum += ms;
                std::snprintf(b, sizeof b, " | %s %.3f (%.1f%%)", stage_name((Stage) i), ms,
                              wm > 0.0 ? 100.0 * ms / wm : 0.0);
                line += b;
            }
            std::snprintf(b, sizeof b, " | stages %.3f (%.1f%%), gap %.3f", sum, wm > 0.0 ? 100.0 * sum / wm : 0.0,
                          wm - sum);
            line += b;
            out.push_back(std::move(line));
        }
        return out;
    }

private:
    std::array<Row, kProfileWidths + 1> rows_{};
};

}  // namespace halo::dense
