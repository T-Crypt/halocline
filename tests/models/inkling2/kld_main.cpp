// tests/models/inkling2/kld_main.cpp - the inkling2 engine against a llama.cpp KL-divergence base (llama-perplexity
// --kl-divergence-base, short or depth): per chunk the tokens before the scored window are prefilled, the window is
// teacher-forced in 16-row decode windows, and every scored row's logits are compared with the base row (the same
// arithmetic as llama-perplexity and include/halo/program/ppl_gate.hpp). Prints mean KLD (+- its standard error),
// top-1 agreement and PPL.   inkling2_kld PACK.halo BASE.kld [CHUNKS] [MAX_CONTEXT]   (env as the engine: HALO_INK_KV)
#include "models/inkling2/model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACK.halo BASE.kld [CHUNKS] [MAX_CONTEXT]\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[2], std::ios::binary);
    char magic[8];
    uint32_t n_ctx = 0;
    int32_t nv[2] = {0, 0}, first = 0;
    f.read(magic, 8);
    f.read((char*) &n_ctx, 4);
    f.read((char*) nv, 8);
    const bool depth = std::memcmp(magic, "_logitsF", 8) == 0;
    if (!f || (!depth && std::memcmp(magic, "_logits_", 8) != 0)) {
        std::fprintf(stderr, "not a KL-divergence base\n");
        return 1;
    }
    if (depth) f.read((char*) &first, 4);
    else first = (int32_t) (n_ctx / 2);
    const int64_t V = nv[0], n_chunk = nv[1], hdr = depth ? 24 : 20, rows = (int64_t) n_ctx - 1 - first;
    const int64_t row_u16 = 2 * ((V + 1) / 2) + 4;
    std::vector<int32_t> tok((size_t) n_chunk * n_ctx);
    f.read((char*) tok.data(), (std::streamsize) tok.size() * 4);
    const int64_t use = argc > 3 ? std::min<int64_t>(std::atoll(argv[3]), n_chunk) : n_chunk;
    halo::model::LoadRequest req;
    req.paths = {argv[1]};
    req.max_context = argc > 4 ? std::atoll(argv[4]) : (int64_t) n_ctx + 64;
    std::string err;
    auto m = halo::models::inkling2::load_model(req, err);
    if (!m) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    if ((int64_t) m->desc().n_vocab != V) {
        std::fprintf(stderr, "the base has %lld tokens, the model %lld\n", (long long) V, (long long) m->desc().n_vocab);
        return 1;
    }
    std::unique_ptr<halo::model::ModelSession> s;
    if (auto r = m->make_session(s); !r.ok()) {
        std::fprintf(stderr, "session: %s\n", r.message.c_str());
        return 1;
    }
    std::vector<uint16_t> base((size_t) row_u16);
    std::vector<float> lg((size_t) V);
    double sk = 0, sk2 = 0, snll = 0, snll_b = 0;
    int64_t n = 0, same = 0;
    for (int64_t c = 0; c < use; ++c) {
        const int32_t* t = tok.data() + c * n_ctx;
        halo::model::PrefillRequest p;
        p.tokens = t;
        p.n_tokens = first;
        p.pos = 0;
        if (auto r = s->prefill_chunk(p); !r.ok()) {
            std::fprintf(stderr, "prefill: %s\n", r.message.c_str());
            return 1;
        }
        for (int64_t at = first; at < (int64_t) n_ctx - 1; at += 16) {
            const int w = (int) std::min<int64_t>(16, (int64_t) n_ctx - 1 - at);
            halo::model::DecodeRequest d;
            std::vector<int32_t> picks((size_t) w);
            d.tokens = t + at;
            d.n_tokens = w;
            d.pos = at;
            d.row_tokens = picks.data();
            if (auto r = s->decode_window(d); !r.ok()) {
                std::fprintf(stderr, "decode: %s\n", r.message.c_str());
                return 1;
            }
            for (int i = 0; i < w; ++i) {
                const int64_t pos = at + i, rr = pos - first;
                s->copy_logits(i, lg.data());
                f.seekg((std::streamoff) (hdr + (int64_t) n_chunk * n_ctx * 4 + (c * rows + rr) * row_u16 * 2));
                f.read((char*) base.data(), (std::streamsize) base.size() * 2);
                float scale, min_lp;
                std::memcpy(&scale, base.data(), 4);
                std::memcpy(&min_lp, base.data() + 2, 4);
                const uint16_t* b = base.data() + 4;
                int64_t top = 0;
                for (int64_t k = 1; k < V; ++k)
                    if (lg[(size_t) k] > lg[(size_t) top]) top = k;
                double sum = 0;
                for (int64_t k = 0; k < V; ++k) sum += std::exp((double) lg[(size_t) k] - lg[(size_t) top]);
                const double lse = lg[(size_t) top] + std::log(sum);
                const int32_t tgt = t[pos + 1];
                double kld = 0;
                int64_t top_b = 0;
                float lb_max = 0;
                for (int64_t k = 0; k < V; ++k) {
                    const float lb = scale * b[k] + min_lp;
                    if (k == 0 || lb > lb_max) lb_max = lb, top_b = k;
                    if (lb > -16.0f) kld += std::exp((double) lb) * (lb - lg[(size_t) k] + lse);
                }
                sk += kld, sk2 += kld * kld, ++n;
                same += top == top_b;
                snll += lse - lg[(size_t) tgt];
                snll_b += -(scale * b[tgt] + min_lp);
            }
        }
        const double mean = sk / n;
        std::fprintf(stderr, "chunk %lld/%lld: mean KLD %.6f  top-1 %.3f%%  PPL %.4f (base %.4f)\n", (long long) c + 1,
                     (long long) use, mean, 100.0 * same / n, std::exp(snll / n), std::exp(snll_b / n));
    }
    const double mean = sk / n, var = sk2 / n - mean * mean;
    std::printf("rows %lld  mean KLD %.6f +- %.6f  top-1 %.3f%%  PPL %.4f  base PPL %.4f\n", (long long) n, mean,
                std::sqrt(var / n), 100.0 * same / n, std::exp(snll / n), std::exp(snll_b / n));
    return 0;
}
