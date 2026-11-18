// tests/model/adapter_builtin_test.cpp - the program's own registration: every adapter the build holds, once each, DeepSeek's
// load error and the flags each model reports as lossy. Linking the models needs the CUDA runtime, so this runs where
// that is installed; no GPU work.
#include "halo/model/adapter.hpp"
#include "halo/model/flags.hpp"
#include "halo/models/builtin.hpp"
#include "halo/program/options.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

int main() {
    using namespace halo;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    // HALO_MODELS decides which adapters exist: each built model is registered once, each left out is refused by name
    size_t built = 0;
    const auto expect = [&](const char* arch, bool is_built) {
        built += is_built ? 1 : 0;
        check((model::find_adapter(arch) != nullptr) == is_built, is_built ? "a built model is registered" : "a model left out is not registered");
        if (!is_built) {
            const std::string why = model::unavailable_architecture(arch);
            check(why.find("left out") != std::string::npos && why.find("HALO_MODELS=") != std::string::npos &&
                      why.find(std::string("-DHALO_MODELS=\"")) != std::string::npos && why.find(arch) != std::string::npos,
                  "a model left out is refused with the option that adds it");
        }
    };
#ifdef HALO_MODEL_QWEN4EXP
    expect("qwen4exp", true);
#else
    expect("qwen4exp", false);
#endif
#ifdef HALO_MODEL_QWEN35
    expect("qwen35", true);
#else
    expect("qwen35", false);
#endif
    check(model::adapter_names().size() == built, "registering twice keeps one adapter per name");
    // DeepSeek was removed from the tree, not left out of this build: it is an unknown architecture
    check(model::find_adapter("deepseek41") == nullptr &&
              model::unavailable_architecture("deepseek41").find("no model adapter is registered") != std::string::npos,
          "deepseek41 is not a known model");
    // the flags as the command line gives them, through the model's own parser and finish_flags
    const auto lossy_of = [](const model::ModelAdapter& a, std::vector<std::string> words) {
        const auto o = a.make_options();
        words.insert(words.begin(), "halo");
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        for (int i = 1; i < (int) argv.size(); ++i) {
            int code = 0;
            model::FlagArgs args((int) argv.size(), argv.data(), i);
            if (!a.parse_flag(argv[i], args, *o, code) || code != 0) return std::vector<program::CommonOptions::LossyFlag>{{"?", argv[i]}};
        }
        (void) a.finish_flags(*o);
        return o->lossy;
    };
    if (const model::ModelAdapter* q = model::find_adapter("qwen4exp")) {
        check(lossy_of(*q, {}).empty(), "qwen4exp with no flag is not lossy");
        const auto one = lossy_of(*q, {"--route-tail-skip", "7"});
        check(one.size() == 1 && one[0].flag == "--route-tail-skip" &&
                  one[0].effect == "--route-tail-skip 7 drops low-ranked missed experts",
              "qwen4exp --route-tail-skip 7 is lossy, with its warning");
        const auto all = lossy_of(*q, {"--relaxed-drafting", "--route-prior", "1.5", "--route-tail-skip"});
        check(all.size() == 3 && all[0].flag == "--relaxed-drafting" && all[1].effect == "--route-prior 1.5 changes expert routing" &&
                  all[2].effect == "--route-tail-skip 9 drops low-ranked missed experts",
              "qwen4exp: every lossy flag, bare ones at their defaults, in the warning order");
    }
    // qwen35 --long-context: the 262K profile (rotated 4-bit K and V), a lossy flag; given flags win, a different --kv
    // is refused. No model file here, so no drafter: the drafter's ring defaults are not set.
    if (const model::ModelAdapter* q = model::find_adapter("qwen35")) {
        const auto finish = [&](std::vector<std::string> words, int& rc) {
            auto o = q->make_options();
            words.insert(words.begin(), "halo");
            std::vector<char*> argv;
            for (auto& w : words) argv.push_back(w.data());
            for (int i = 1; i < (int) argv.size(); ++i) {
                int code = 0;
                model::FlagArgs args((int) argv.size(), argv.data(), i);
                if (!q->parse_flag(argv[i], args, *o, code)) {
                    if (std::string(argv[i]) == "--max-context") {   // a common flag: the program's parser sets it
                        o->max_context = std::atoll(args.next("--max-context"));
                        o->max_context_given = true;
                    } else {
                        rc = -1;
                        return o;
                    }
                }
            }
            const auto r = q->finish_flags(*o);
            rc = r ? *r : 0;
            return o;
        };
        int rc = 0;
        auto plain = finish({}, rc);
        check(rc == 0 && plain->kv.empty() && plain->max_context == program::kDefaultContext && plain->lossy.empty(),
              "qwen35 without --long-context keeps its default cache and context and is not lossy");
        auto lc = finish({"--long-context"}, rc);
        check(rc == 0 && lc->kv == "q4_0" && lc->max_context == 262144 && lc->lossy.size() == 1 &&
                  lc->lossy[0].flag == "--long-context",
              "qwen35 --long-context: --kv q4_0, 262,144 cells, reported lossy");
        auto mc = finish({"--long-context", "--max-context", "200000"}, rc);
        check(rc == 0 && mc->kv == "q4_0" && mc->max_context == 200000, "qwen35 --long-context keeps a given --max-context");
        auto same = finish({"--kv", "q4_0", "--long-context"}, rc);
        check(rc == 0 && same->kv == "q4_0", "qwen35 --long-context with --kv q4_0 is the same choice");
        (void) finish({"--long-context", "--kv", "k8v4"}, rc);
        check(rc == 2, "qwen35 --long-context refuses a different --kv");
    }
    return failures ? 1 : 0;
}
