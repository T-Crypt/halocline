// tests/models/qwen35/mtp_levers_test.cpp - the dense drafter's cache levers (--mtp-kv, --mtp-window), host only: the
// values the flags take, the ring's size against the read span a pass writes over, the planner's bytes per format, and
// the image check that refuses another format or a ring. The mapping itself (`pos % cells`) is the attention kernel's;
// the ring test runs it here against the same rule the kernel documents, so the constant is checked, not the kernel.
#include "halo/dense/model.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// The kernel's rule, cell by cell: position p lives at p % cells; row t of a pass at p0 reads [max(0, p0 + t - N + 1),
// p0 + t], and the pass writes p0 .. p0 + T - 1 before its rows read them. True when every read finds the position it
// asked for: the ring's cells before the pass are the last write of each slot over positions 0 .. p0 - 1.
bool ring_holds(int64_t window, int64_t cells, int64_t p0, int T) {
    std::vector<int64_t> slot((size_t) cells, -1);
    for (int64_t p = 0; p < p0; ++p) slot[(size_t) (p % cells)] = p;
    for (int64_t p = p0; p < p0 + T; ++p) slot[(size_t) (p % cells)] = p;
    for (int t = 0; t < T; ++t) {
        const int64_t hi = p0 + t, lo = std::max<int64_t>(0, hi - window + 1);
        for (int64_t q = lo; q <= hi; ++q)
            if (slot[(size_t) (q % cells)] != q) return false;
    }
    return true;
}

double mib(uint64_t bytes) { return (double) bytes / 1048576.0; }
}  // namespace

int main() {
    using namespace halo::dense;
    // --mtp-kv's values
    KvFormat f = KvFormat::K8V4;
    check(mtp_kv_format("", f) && f == KvFormat::Fp16, "--mtp-kv unset is fp16");
    check(mtp_kv_format("fp16", f) && f == KvFormat::Fp16, "--mtp-kv fp16");
    check(mtp_kv_format("int8", f) && f == KvFormat::Int8, "--mtp-kv int8");
    check(!mtp_kv_format("q4", f), "--mtp-kv q4 is refused");
    check(!mtp_kv_format("INT8", f), "--mtp-kv is case-sensitive");
    check(!mtp_kv_format("k8v4", f), "--mtp-kv k8v4 is refused: the main cache's format");
    // the conversation cache's numbers, for the main cache and the drafter alike (a store from either branch reads back
    // under the same flags)
    check(kv_format_id(KvFormat::Fp16) == 1 && kv_format_id(KvFormat::Int8) == 2 && kv_format_id(KvFormat::K8V4) == 3,
          "a stored format is 1 fp16, 2 int8, 3 K8V4");

    // the levers against the drafter the load picks
    check(mtp_option_error(false, "", 0, Drafter::Mtp).empty(), "no levers on the MTP drafter");
    check(mtp_option_error(true, "fp16", 0, Drafter::Mtp).empty(), "--mtp-kv fp16 on the MTP drafter");
    check(mtp_option_error(true, "int8", 0, Drafter::Mtp).empty(), "--mtp-kv int8 on the MTP drafter");
    check(mtp_option_error(false, "", 32768, Drafter::Mtp).empty(), "--mtp-window on the MTP drafter");
    check(mtp_option_error(true, "int8", 32768, Drafter::Mtp).empty(), "both levers together");
    check(has(mtp_option_error(true, "q4", 0, Drafter::Mtp), "--mtp-kv q4"), "a bad format names the flag and its values");
    check(has(mtp_option_error(true, "k8v4", 0, Drafter::Mtp), "the main cache's format"),
          "--mtp-kv k8v4 is refused by name: the drafter keeps fp16 or int8");
    check(has(mtp_option_error(false, "", -1, Drafter::Mtp), "--mtp-window -1"), "a negative window is refused");
    check(has(mtp_option_error(false, "", (int64_t) 1 << 41, Drafter::Mtp), "--mtp-window"),
          "a window past any context is refused");
    check(has(mtp_option_error(true, "int8", 0, Drafter::Block), "needs the pack's MTP drafter"),
          "--mtp-kv on the block drafter is refused");
    check(has(mtp_option_error(false, "", 32768, Drafter::Block), "--mtp-window"),
          "--mtp-window on the block drafter names itself");
    check(has(mtp_option_error(true, "fp16", 0, Drafter::None), "needs the pack's MTP drafter"),
          "--mtp-kv without a drafter is refused, even at its default");
    check(mtp_option_error(false, "", 0, Drafter::None).empty(), "no levers without a drafter is fine");

    // the ring: window + kGqaMaxWindow - 1 cells (at least the kernel's 32-cell chunk), and that constant is what a
    // pass of 16 rows needs
    check(mtp_ring_cells(1) == 32 && mtp_ring_cells(17) == 32 && mtp_ring_cells(18) == 33, "a small window's ring is one sweep chunk");
    check(mtp_ring_cells(1024) == 1039 && mtp_ring_cells(32768) == 32783, "the ring is the window plus 15 cells");
    check(mtp_ring_cap(262144, 0) == 0, "no window: cells at their own positions");
    check(mtp_ring_cap(262144, 32768) == 32783, "a 32768 window in a 262144 context is a ring of 32783");
    check(mtp_ring_cap(32783, 32768) == 0, "a window as wide as the context keeps absolute cells");
    check(mtp_ring_cap(262144, 262144) == 0, "a window at the context is no ring");
    check(mtp_ring_cap(0, 32768) == 0, "no context: nothing to ring");
    for (const int64_t window : {int64_t{1}, int64_t{16}, int64_t{1024}, int64_t{32768}}) {
        const int64_t cells = mtp_ring_cells(window);
        for (const int64_t p0 : {int64_t{0}, int64_t{1}, window / 2, window, window + 15, 2 * window + 7, 5 * window}) {
            check(ring_holds(window, cells, p0, 16), "a pass of 16 rows at " + std::to_string((long long) p0) + " reads what it wrote (window " +
                                                         std::to_string((long long) window) + ")");
        }
    }
    // one cell less fails where the formula is exact (a floored window's ring has slack, so only these are tight)
    for (const int64_t window : {int64_t{17}, int64_t{1024}, int64_t{32768}})
        check(!ring_holds(window, mtp_ring_cells(window) - 1, window + 15, 16),
              "one cell less does not (window " + std::to_string((long long) window) + ")");

    // the planner: one cell of the Qwen3.8-27B drafter, and the drafter's bytes at a 262144-cell context
    Geometry g;
    g.n_kv = 4;
    g.head_dim = 256;
    check(kv_cell_bytes(g, KvFormat::Fp16) == 4096, "an fp16 drafter cell is 4 * 256 * 2 * 2 bytes");
    check(kv_cell_bytes(g, KvFormat::Int8) == 2112, "an int8 drafter cell is 4 * (256 + 8) * 2 bytes");
    check(mib(kv_cell_bytes(g, KvFormat::Fp16) * 262144) == 1024.0, "fp16 at 262144 cells: 1024 MiB");
    check(mib(kv_cell_bytes(g, KvFormat::Int8) * 262144) == 528.0, "int8 at 262144 cells: 528 MiB");
    check(kv_cell_bytes(g, KvFormat::Fp16) * 32783 == 134279168, "fp16 at a 32768 window's ring: 134279168 B (128.05 MiB)");
    check(kv_cell_bytes(g, KvFormat::Int8) * 32783 == 69237696, "int8 at a 32768 window's ring: 69237696 B (66.03 MiB)");

    // the drafter's part of a conversation image: the same format, the matched rows, and no ring
    Options o;
    o.max_context = 4096;
    check(mtp_image_matches(o, 1, 100, 100), "an fp16 image at its rows is this session's");
    check(!mtp_image_matches(o, 2, 100, 100), "an int8 image is not an fp16 session's");
    check(!mtp_image_matches(o, 1, 99, 100), "an image shorter than the match is refused");
    o.mtp_format = KvFormat::Int8;
    check(mtp_image_matches(o, 2, 100, 100), "an int8 image is an int8 session's");
    check(!mtp_image_matches(o, 1, 100, 100), "an fp16 image is not an int8 session's");
    o.mtp_window = 32768;
    o.max_context = 262144;   // the window's ring is smaller than the context: no image
    check(!mtp_image_matches(o, 2, 100, 100), "a windowed session takes no image (its cells are a ring)");

    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
