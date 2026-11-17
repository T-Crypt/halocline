// tests/models/qwen35/mtp_cache_test.cpp - the dense drafter's cache levers end to end on a 27B pack: with --mtp-kv int8
// and with --mtp-window 32768 the emitted greedy tokens and every window's first-row logits are the default run's, over a
// prompt longer than the window; each arm prints how many drafts the verifier took. The drafter's cache only feeds the
// drafts: the verify decides every token, so a different cache may change the drafts and the acceptance, never the
// output. Needs a pack (argument or HALO_QWEN35_PACK) and its GPU; exits 77 without either.
#include "halo/core/model_file.hpp"
#include "halo/model/adapter.hpp"
#include "models/qwen35/adapter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace halo;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// One run's output: the emitted greedy tokens, and a hash of the logits of every window's first row, keyed by the
// position that row predicted from (identical in every arm when the output is).
struct Run {
    std::vector<int32_t> tokens;
    std::vector<int32_t> drafts;   // every draft the drafter offered, in order (bit-identical runs make the same ones)
    std::map<int64_t, uint64_t> rows;
    int64_t offered = 0, accepted = 0, windows = 0;
};

uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// The prompt read, then greedy windows of the session's own width: the drafter's drafts fill the window, the first draft
// the verify does not confirm ends it, and the target's own row 0 leaves.  The engine's loop (src/program/generate/
// adapter.cpp) over the session, without the server.
Run run_arm(const model::ModelAdapter& a, const std::string& pack, const std::vector<int32_t>& prompt, int n_new,
            const char* mtp_kv, int64_t mtp_window) {
    Run out;
    model::LoadRequest req;
    req.paths = {pack};
    req.max_context = (int64_t) prompt.size() + n_new + 64;
    req.draft = true;
    req.spec = 16;
    req.kv = "int8";   // every arm: the main cache is not what this test compares
    req.mtp_kv = mtp_kv;
    req.mtp_window = mtp_window;
    std::unique_ptr<model::LoadedModel> loaded;
    if (const model::Result e = a.load(req, loaded); !e.ok() || !loaded) {
        std::fprintf(stderr, "load: %s\n", e.message.c_str());
        std::exit(1);
    }
    std::unique_ptr<model::ModelSession> s;
    if (const model::Result e = loaded->make_session(s); !e.ok() || !s->prepare().ok()) {
        std::fprintf(stderr, "session: %s\n", e.message.c_str());
        std::exit(1);
    }
    const int cap = s->max_window();
    if (cap < 2) {
        std::fprintf(stderr, "the pack has no MTP drafter (max_window %d)\n", cap);
        std::exit(1);
    }
    const std::string info = s->info_words();
    if (std::string(mtp_kv) == "int8") check(info.find("mtp_kv=int8") != std::string::npos, "the run says --mtp-kv int8");
    if (mtp_window > 0) {
        check(info.find("mtp_window=" + std::to_string((long long) mtp_window)) != std::string::npos,
              "the run says its window");
        if (mtp_window + 15 < req.max_context)   // else the window reaches the context: cells stay absolute
            check(info.find("mtp_ring=" + std::to_string((long long) mtp_window + 15)) != std::string::npos,
                  "the run says the ring's cells");
    }
    model::PrefillRequest pr;
    pr.tokens = prompt.data();
    pr.n_tokens = (int) prompt.size() - 1;
    pr.total_tokens = pr.n_tokens;
    if (const model::Result e = s->prefill_chunk(pr); !e.ok()) {
        std::fprintf(stderr, "prefill: %s\n", e.message.c_str());
        std::exit(1);
    }
    int32_t next = prompt.back();   // the prompt's last token is the first window's own row
    int64_t pos = (int64_t) prompt.size() - 1;
    std::vector<int32_t> window((size_t) cap), rows((size_t) cap), drafts((size_t) cap);
    std::vector<float> conf((size_t) cap);
    std::vector<float> logits((size_t) loaded->desc().n_vocab);
    while ((int64_t) out.tokens.size() < n_new) {
        int n_draft = 0;
        model::Result d = s->draft_begin(next, pos, cap - 1);
        if (d.ok()) d = s->draft_end(drafts.data(), conf.data(), cap - 1, n_draft);
        if (!d.ok() && d.status != model::Status::NotImplemented) {
            std::fprintf(stderr, "draft: %s\n", d.message.c_str());
            std::exit(1);
        }
        const int T = 1 + n_draft;
        window[0] = next;
        std::copy(drafts.begin(), drafts.begin() + n_draft, window.begin() + 1);
        model::DecodeRequest dr;
        dr.tokens = window.data();
        dr.n_tokens = T;
        dr.pos = pos;
        (T > 1 ? dr.row_tokens : dr.next_token) = rows.data();
        if (const model::Result e = s->decode_window(dr); !e.ok()) {
            std::fprintf(stderr, "decode: %s\n", e.message.c_str());
            std::exit(1);
        }
        int a = 0;
        while (a < T - 1 && window[(size_t) a + 1] == rows[(size_t) a]) ++a;
        out.offered += T - 1;
        out.accepted += a;
        ++out.windows;
        out.drafts.insert(out.drafts.end(), drafts.begin(), drafts.begin() + n_draft);
        // row 0 saw the accepted prefix alone: its logits are every arm's, whatever drafts filled the other rows
        if (const model::Result e = s->copy_logits(0, logits.data()); !e.ok()) {
            std::fprintf(stderr, "copy_logits: %s\n", e.message.c_str());
            std::exit(1);
        }
        out.rows[pos] = fnv1a(logits.data(), logits.size() * sizeof(float));
        for (int i = 0; i <= a && (int64_t) out.tokens.size() < n_new; ++i) out.tokens.push_back(rows[(size_t) i]);
        pos += a + 1;
        next = rows[(size_t) a];
    }
    return out;
}

void compare(const Run& base, const Run& arm, const std::string& what, bool same_drafts) {
    check(arm.tokens == base.tokens, what + ": the same greedy tokens");
    bool same = arm.rows.size() == base.rows.size();
    for (const auto& [pos, h] : arm.rows) {
        const auto it = base.rows.find(pos);
        same = same && it != base.rows.end() && it->second == h;
    }
    check(same, what + ": the same first-row logits at every window");
    if (same_drafts) check(arm.drafts == base.drafts, what + ": the same drafts (the window masks nothing here)");
}
}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("HALO_QWEN35_PACK");
    const std::string pack = argc > 1 ? argv[1] : env != nullptr ? env : "";
    int devices = 0;
    if (pack.empty() || cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("skip: no 27B pack (argument or HALO_QWEN35_PACK) or no GPU\n");
        return 77;
    }
    // the pack's GGUF view, as the program opens it
    core::ModelFile file;
    std::string err;
    if (!file.open(pack, err) || !file.load_meta(err) || !file.register_gguf_view(err)) {
        std::fprintf(stderr, "%s: %s\n", pack.c_str(), err.c_str());
        return 1;
    }
    const std::unique_ptr<model::ModelAdapter> a = models::qwen35::make_adapter();
    // First a prompt shorter than the window: there the window masks nothing, so the drafts themselves must be the
    // default run's (a cheap check of the window's plumbing before the long read).  A prompt of ordinary token ids.
    const int n_short = 2048;
    std::vector<int32_t> short_prompt((size_t) n_short);
    for (int i = 0; i < n_short; ++i) short_prompt[(size_t) i] = 1000 + (int32_t) ((int64_t) i * 7919 % 30000);
    const Run short_base = run_arm(*a, pack, short_prompt, 16, "", 0);
    const Run short_win = run_arm(*a, pack, short_prompt, 16, "", 32768);
    compare(short_base, short_win, "--mtp-window 32768 under its window", true);
    // then a prompt longer than the window: the ring wraps, and only the output must be the default run's
    const int n_prompt = 36000;
    std::vector<int32_t> prompt((size_t) n_prompt);
    for (int i = 0; i < n_prompt; ++i) prompt[(size_t) i] = 1000 + (int32_t) ((int64_t) i * 7919 % 30000);
    const Run base = run_arm(*a, pack, prompt, 48, "", 0);
    const Run i8 = run_arm(*a, pack, prompt, 48, "int8", 0);
    const Run win = run_arm(*a, pack, prompt, 48, "", 32768);
    const auto report = [](const char* name, const Run& r) {
        std::printf("%s: %lld drafts offered, %lld accepted (%.1f%%), %lld windows\n", name, (long long) r.offered,
                    (long long) r.accepted, r.offered > 0 ? 100.0 * (double) r.accepted / (double) r.offered : 0.0,
                    (long long) r.windows);
    };
    report("short prompt, default", short_base);
    report("short prompt, --mtp-window 32768", short_win);
    report("default", base);
    report("--mtp-kv int8", i8);
    report("--mtp-window 32768", win);
    compare(base, i8, "--mtp-kv int8", false);
    compare(base, win, "--mtp-window 32768", false);
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
