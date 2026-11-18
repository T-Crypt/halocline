// tests/models/qwen35/draft_vocab_test.cpp - --draft-vocab's file and its two numbers, host only: the reader's format
// (strictly ascending int32 ids inside the vocabulary, every other shape a named refusal), the flag's rules against the
// drafter the load picks, the row map a load builds from the file, and the bytes the planner counts for the head's rows
// and the shrunk logits row.  No card: the gather itself is tests/models/qwen35/draft_head_test.cpp.
#include "halo/dense/model.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

// a file of little-endian int32 ids
std::string write_ids(const std::string& name, const std::vector<int32_t>& ids) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(ids.data()), (std::streamsize) ids.size() * 4);
    f.close();
    return p.string();
}
std::string write_bytes(const std::string& name, const std::vector<uint8_t>& v) {
    const std::filesystem::path p = std::filesystem::temp_directory_path() / name;
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize) v.size());
    f.close();
    return p.string();
}
}  // namespace

int main() {
    using namespace halo::dense;
    // the reader: a strictly ascending list inside the vocabulary
    {
        std::vector<int32_t> ids;
        std::string err;
        const std::string ok = write_ids("qwen35_draft_vocab_ok.bin", {0, 3, 7, 42, 248319});
        check(read_draft_vocab(ok, 248320, ids, err), "a sorted list of ids reads: " + err);
        check(ids == std::vector<int32_t>({0, 3, 7, 42, 248319}), "and comes back in the file's order");
        // the row map a load builds: row r is ids[r], and ids[r]'s position among the ids is r (no duplicate, no gap lost)
        bool map_ok = true;
        for (size_t r = 0; r < ids.size(); ++r) {
            const auto at = std::lower_bound(ids.begin(), ids.end(), ids[r]) - ids.begin();
            map_ok = map_ok && at == (long) r;
        }
        check(map_ok, "the row -> id map is its own inverse lookup");
        std::remove(ok.c_str());
    }
    // every other shape is a refusal that names the file or the row
    {
        std::vector<int32_t> ids;
        std::string err;
        err.clear();
        check(!read_draft_vocab((std::filesystem::temp_directory_path() / "qwen35_draft_vocab_missing.bin").string(), 8, ids, err),
              "a missing file is refused");
        check(has(err, "cannot open"), "and the error names the open");

        const std::string not4 = write_bytes("qwen35_draft_vocab_5.bin", {1, 0, 0, 0, 2});
        check(!read_draft_vocab(not4, 8, ids, err), "a file of 5 bytes is refused");
        check(has(err, "5 bytes"), "and the error names the size");
        std::remove(not4.c_str());

        const std::string empty = write_bytes("qwen35_draft_vocab_empty.bin", {});
        check(!read_draft_vocab(empty, 8, ids, err), "an empty file is refused");
        std::remove(empty.c_str());

        const std::string neg = write_ids("qwen35_draft_vocab_neg.bin", {-1, 2});
        check(!read_draft_vocab(neg, 8, ids, err), "a negative id is refused");
        check(has(err, "outside [0, 8)"), "and the error names the vocabulary's range");
        std::remove(neg.c_str());

        const std::string past = write_ids("qwen35_draft_vocab_past.bin", {2, 8});
        check(!read_draft_vocab(past, 8, ids, err), "an id at the vocabulary's size is refused");
        std::remove(past.c_str());

        const std::string dup = write_ids("qwen35_draft_vocab_dup.bin", {2, 2});
        check(!read_draft_vocab(dup, 8, ids, err), "a repeated id is refused");
        check(has(err, "strictly ascending"), "and the error says the list must be strictly ascending");
        std::remove(dup.c_str());

        const std::string down = write_ids("qwen35_draft_vocab_down.bin", {5, 4});
        check(!read_draft_vocab(down, 8, ids, err), "a descending pair is refused");
        std::remove(down.c_str());
    }

    // --draft-vocab's rules: the MTP drafter only (the block drafter already scores a subset of its own)
    check(draft_vocab_option_error(false, Drafter::None).empty(), "no flag, no error");
    check(draft_vocab_option_error(true, Drafter::Mtp).empty(), "--draft-vocab on the MTP drafter");
    check(has(draft_vocab_option_error(true, Drafter::Block), "needs the pack's MTP drafter"),
          "--draft-vocab on the block drafter is refused by name");
    check(has(draft_vocab_option_error(true, Drafter::None), "needs the pack's MTP drafter"),
          "--draft-vocab without a drafter is refused");

    // the head's bytes and the logits row's: a Q6_K head row of 5120 values is 20 blocks of 210 bytes
    Geometry g;
    g.n_vocab = 248320;
    g.n_embd = 5120;
    g.n_ff = 17408;
    Linear head;
    head.type = 14;   // Q6_K
    head.rows = g.n_vocab;
    head.cols = g.n_embd;
    check(draft_head_bytes(head, 1) == 4200, "a Q6_K head row is 4200 bytes");
    check(draft_head_bytes(head, 131072) == 550502400, "131072 of them: 550502400 B (525.0 MiB)");
    check(draft_head_bytes(head, 0) == 0, "no rows, no bytes");

    // the planner: the drafter's logits row shrinks by the rows the subset drops, and nothing else moves
    const uint64_t full = session_bytes(g, 4096, true, KvFormat::Fp16, 0, KvFormat::Fp16, 0, 0);
    const uint64_t half = session_bytes(g, 4096, true, KvFormat::Fp16, 0, KvFormat::Fp16, 0, 131072);
    check(full > half && full - half == (uint64_t) (g.n_vocab - 131072) * 4,
          "a 131072-row draft head takes (248320 - 131072) * 4 bytes off the session's logits row");
    check(session_bytes(g, 4096, false, KvFormat::Fp16, 0, KvFormat::Fp16, 0, 131072) ==
              session_bytes(g, 4096, false, KvFormat::Fp16, 0, KvFormat::Fp16, 0, 0),
          "without the MTP drafter the draft rows count for nothing");

    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
