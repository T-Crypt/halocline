// tests/dense/profile_report_test.cpp - --profile's table: the per-width aggregation, each stage's mean ms and its
// share of the window, the sum of the stages against the window's wall, and the report's text (host only, no card).
#include "halo/dense/profile.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace halo::dense;

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// the line for a width (the first that starts with it), or ""
std::string line_for(const std::vector<std::string>& lines, int t) {
    const std::string want = "T " + std::to_string(t) + " ";
    for (const std::string& l : lines)
        if (l.rfind(want, 0) == 0) return l;
    return {};
}

void add_window(ProfileTable& t, int width, double wall_ms,
                std::initializer_list<std::pair<Stage, double>> stages) {
    double ms[kStageCount] = {};
    for (const auto& [s, v] : stages) ms[(int) s] += v;
    t.add(width, wall_ms, ms);
}

}  // namespace

int main() {
    ProfileTable table;
    check(table.report().empty(), "an empty table prints nothing");
    check(table.windows() == 0, "an empty table has no windows");

    // two windows of T=2, then one of T=1
    add_window(table, 2, 10.0, {{Stage::AttnPre, 2.0}, {Stage::FfnUp, 1.0}, {Stage::Sample, 0.5}});
    add_window(table, 2, 8.0, {{Stage::AttnPre, 4.0}, {Stage::Head, 1.0}});
    add_window(table, 1, 5.0, {{Stage::GdnRec, 1.0}});
    check(table.windows() == 3, "three windows over two widths");

    const std::vector<std::string> lines = table.report();
    check(lines.size() == 3, "a header and one line per width with windows");
    const std::string& head = lines[0];
    check(has(head, "3 decode windows"), "the header counts the windows");
    check(has(head, "uncaptured"), "the header says the window ran uncaptured");
    check(has(head, "launch gap"), "the header says a stage's ms holds its launch gap");
    check(has(head, "host-side"), "the header says accept is host-side");
    check(has(head, "gap"), "the header names the gap column");

    // T=2: mean window 9.000; attn-pre 3.000 (33.3%), ffn-up 0.500 (5.6%), head 0.500 (5.6%),
    // sample 0.250 (2.8%); the stages sum to 4.250 (47.2%) and the gap is 4.750
    const std::string t2 = line_for(lines, 2);
    check(t2 == "T 2 windows 2 window 9.000 ms | attn-pre 3.000 (33.3%) | ffn-up 0.500 (5.6%) | head 0.500 (5.6%) | "
                "sample 0.250 (2.8%) | stages 4.250 (47.2%), gap 4.750",
          "T=2's line: the means, the shares, the stage sum and the gap");
    const std::string t1 = line_for(lines, 1);
    check(t1 == "T 1 windows 1 window 5.000 ms | gdn-rec 1.000 (20.0%) | stages 1.000 (20.0%), gap 4.000",
          "T=1's line: one stage, its share, the sum and the gap");
    check(line_for(lines, 3).empty(), "a width with no window has no line");
    check(!has(t2, "attn-qk"), "a stage with no time is left out");

    // a mark pool overflow counts in the header
    {
        const std::vector<std::string> s = table.report(4);
        check(has(s[0], "4 skipped"), "the header counts the windows skipped for a full mark pool");
        check(!has(table.report()[0], "skipped"), "a run without a skipped window says nothing about them");
    }

    // a width outside the window is dropped, and a negative stage time counts as zero
    {
        ProfileTable t;
        add_window(t, 17, 1.0, {{Stage::AttnPre, 1.0}});
        add_window(t, 0, 1.0, {{Stage::AttnPre, 1.0}});
        check(t.windows() == 0, "widths 0 and 17 are not decode windows and are dropped");
        add_window(t, 3, 2.0, {{Stage::AttnCore, -1.0}});
        check(t.windows() == 1, "a valid width is added");
        const std::vector<std::string> l = t.report();
        check(line_for(l, 3) == "T 3 windows 1 window 2.000 ms | stages 0.000 (0.0%), gap 2.000",
              "a clock that ran backwards counts as zero and leaves the stage out");
        t.clear();
        check(t.report().empty() && t.windows() == 0, "clear empties the table and its text");
    }

    // the column names the report prints
    check(std::string(stage_name(Stage::AttnPre)) == "attn-pre", "attn-pre is named");
    check(std::string(stage_name(Stage::FfnDown)) == "ffn-down", "ffn-down is named");
    check(std::string(stage_name(Stage::MtpDraft)) == "mtp-draft", "mtp-draft is named");
    check(std::string(stage_name(Stage::MtpHist)) == "mtp-hist", "mtp-hist is named");
    check(std::string(stage_name(Stage::None)) == "-", "a boundary is not a column");
    check(std::string(stage_name(Stage::Count)) == "-", "the end marker is not a column");

    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
