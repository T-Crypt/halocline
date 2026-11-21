// tests/models/inkling2/greedy_main.cpp - first-light check of the inkling2 forward: load a pack, read a prompt of token
// ids, decode N tokens greedily and print them (and the time per decoded token). Compared with llama.cpp's greedy
// continuation of the same ids (llama-server /completion, temperature 0) by tools/inkling2/greedy_vs_llama.py.
//   inkling2_greedy PACK.halo IDS.txt N [MAX_CONTEXT]
#include "models/inkling2/model.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s PACK.halo IDS.txt N [MAX_CONTEXT]\n", argv[0]);
        return 2;
    }
    std::vector<int32_t> ids;
    {
        std::ifstream f(argv[2]);
        long long v;
        while (f >> v) ids.push_back((int32_t) v);
    }
    const int n = std::atoi(argv[3]);
    halo::model::LoadRequest req;
    req.paths = {argv[1]};
    req.max_context = argc > 4 ? std::atoll(argv[4]) : 32768;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    auto m = halo::models::inkling2::load_model(req, err);
    if (!m) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<halo::model::ModelSession> s;
    if (auto r = m->make_session(s); !r.ok()) {
        std::fprintf(stderr, "session: %s\n", r.message.c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    halo::model::PrefillRequest p;
    p.tokens = ids.data();
    p.n_tokens = (int) ids.size() - 1;
    p.pos = 0;
    if (p.n_tokens > 0)
        if (auto r = s->prefill_chunk(p); !r.ok()) {
            std::fprintf(stderr, "prefill: %s\n", r.message.c_str());
            return 1;
        }
    auto t2 = std::chrono::steady_clock::now();
    int32_t tok = ids.back();
    int64_t pos = (int64_t) ids.size() - 1;
    std::vector<int32_t> out;
    auto t3 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        halo::model::DecodeRequest d;
        int32_t next = -1;
        d.tokens = &tok;
        d.n_tokens = 1;
        d.pos = pos;
        d.next_token = &next;
        if (auto r = s->decode_window(d); !r.ok()) {
            std::fprintf(stderr, "decode: %s\n", r.message.c_str());
            return 1;
        }
        out.push_back(next);
        tok = next;
        ++pos;
        if (i == 0) t3 = std::chrono::steady_clock::now();
    }
    auto t4 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? " " : "", out[i]);
    std::printf("\n");
    std::fprintf(stderr, "load %.1f s, prefill %d tokens %.1f ms (%.0f t/s), decode %d tokens %.1f ms (%.1f t/s after the first)\n",
                 ms(t0, t1) / 1000.0, p.n_tokens, ms(t1, t2), p.n_tokens / (ms(t1, t2) / 1000.0), n, ms(t2, t4),
                 n > 1 ? (n - 1) / (ms(t3, t4) / 1000.0) : 0.0);
    return 0;
}
