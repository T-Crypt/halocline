// src/models/qwen4exp/serve/qfn_decode.cpp - Flash-Next's side of the window loop: verify windows, drafts, adaptive rounds.
#include "basalt/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "../../../program/serve/decode_loop.hpp"

namespace basalt::models::qwen4exp::serve {

namespace {
// --draft-log FILE: one line per window (T, accepted, draft probabilities, ms) for draft-policy study
FILE* draft_log_file() {
    static FILE* const f = basalt::core::diagnostics().draft_log.empty() ? nullptr
        : std::fopen(basalt::core::diagnostics().draft_log.c_str(), "a");
    return f;
}
// spec_log: per window, the target's sampling distribution at each verified row beside the draft head's top-4
// of the draft that row checks, for offline tree-draft studies (written next to --draft-log as FILE.tk)
FILE* tk_log_file() {
    static FILE* const f = basalt::core::exp_flag("spec_log", 0) != 0 && !basalt::core::diagnostics().draft_log.empty()
        ? std::fopen((basalt::core::diagnostics().draft_log + ".tk").c_str(), "a") : nullptr;
    return f;
}
}  // namespace

QfnRequest::DecSnap QfnRequest::dec_snap() {
    return DecSnap{runtime_.ver.ms_wait, runtime_.ver.ms_pool, runtime_.ver.ms_host, runtime_.drive.d.ms_plan, runtime_.drive.d.ms_actq, runtime_.drive.d.ms_jobs,
                   runtime_.drive.d.ms_run,
                   runtime_.drive.d.multi_misses, runtime_.drive.d.multi_entries, runtime_.drive.d.cache_hits, runtime_.drive.d.pcie_experts};
}

Flow QfnRequest::decode_begin() {
    if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }   // the windows between prompt chunks swapped it
    return Flow::proceed;
}

void QfnRequest::seed_tiers() {
    {   // a request decoded here starts its adaptation from the experts its own prompt routed to
        std::vector<uint32_t> hist;
        int64_t hist_tokens = 0;
        const bool seeded = !runtime_.drive.d.usage.empty() && runtime_.sp.route_hist_take(hist, hist_tokens) && (!in_slot || solo_req);
        if (seeded) seed_usage_from_prompt(runtime_.drive.d.usage, hist, hist_tokens, runtime_.host_res);
    }
}

void QfnRequest::decode_state_begin() {
    drafts = std::vector<int32_t>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0);
    dprob = std::vector<float>((size_t) std::max(runtime_.S, basalt::core::kMtpChainMax), 0.0f);
    sbuf = std::vector<int32_t>((size_t) runtime_.S, 0);
    if (runtime_.o.suffix_draft > 0) {
        runtime_.sfx->reset();
        for (int64_t t : ids) runtime_.sfx->append((int32_t) t);
    }
    sfx_windows = 0;
    sfx_drafts = 0;
    sfx_ok = 0;
    cv = &runtime_.ver;
    n_chain = 0;                       // drafts the last chain produced
}

void QfnRequest::decode_stats_begin() {
    // --decode-timing: where a request's decode time goes (one line per request)
    dec_timing = basalt::core::diagnostics().decode_timing;

    ds0 = dec_snap();
    dt_run = 0;
    dt_commit = 0;
    dt_draft = 0;
    dec_windows = 0;
    dec_T = 0;
    decode_hits0 = runtime_.drive.d.cache_hits;
    swapped0 = runtime_.xq_swapped;
    noplace0 = runtime_.xq_noplace;
    // the RAM and file tiers of this request (the mmap source; 0 with the arena)
    ram0 = runtime_.src.ram_reads();
    files0 = runtime_.src.file_reads();
    file_bytes0 = runtime_.src.file_read_bytes();
    decode_look0 = runtime_.drive.d.cache_hits + runtime_.drive.d.cache_admitted + runtime_.drive.d.cache_refused;
    offload0 = runtime_.drive.d.offload_entries;
    adapt_age = 2;   // loop tops since the last adaptive round
    applied_early = false;

    copy_live = false;
    copy_done = false;   // exchanges applied early whose RAM copies are under way
    draft_log = draft_log_file();
    if (draft_log != nullptr) {
        std::fprintf(draft_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp);
        std::fflush(draft_log);
    }
    tk_log = tk_log_file();
    tk_valid = false;
    if (tk_log != nullptr) { std::fprintf(tk_log, "R temp=%g S=%d\n", (double) req_temperature, runtime_.S_mtp); std::fflush(tk_log); }
}

inline int QfnRequest::draft_window() {
    int T = 1;
        while (T < runtime_.S_mtp && basalt::core::draft_keep(dprob.data(), T - 1, cum_adapt.th())) ++T;
    return T;
}

inline void QfnRequest::lookup_window(int& T, bool& from_sfx, int& sfx_match) {
        if (runtime_.o.suffix_draft > 0 && !first_window && ppl == nullptr) {
            const int k = runtime_.sfx->propose(runtime_.S - 1, sbuf.data());
            sfx_match = runtime_.sfx->last_match();
            if (k > 0 && sbuf[0] == drafts[0]) {
                const basalt::spec::DraftPolicy::Pick pk = runtime_.policy->choose(T, k, sfx_match);
                if (pk.lookup) { T = pk.t; from_sfx = true; }
            }
        }
}

inline void QfnRequest::fill_drafts(int T, bool from_sfx) {
        for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
}

inline int QfnRequest::before_window(int T, int64_t) {
        runtime_.drive.d.layers = 0;
        runtime_.drive.d.experts = 0;
        runtime_.drive.d.failed = false;
        // the previous adapt round's copies land first: with a non-blocking query, whether a swapped-in expert
        // ran on the GPU or the CPU (they round differently) would depend on the copy's timing
        basalt::core::decode_gaps().mark(17);
        // an adaptive round's swaps land before the second window after it (a fixed window, so reproducible),
        // applied beside the draft before it (apply_thr) when their copies are done
        if (apply_thr.joinable()) apply_thr.join();
        if (copy_live && copy_done) {   // the RAM copies are done: their bookkeeping, between windows
            basalt::core::decode_events().tag(1u << 3);
            runtime_.src.commit_exchanges();
            copy_live = false;
        }
        ++adapt_age;
        if (adapt_age >= 2 && !(applied_early && adapt_age == 2)) {
            if (!runtime_.pending.empty()) basalt::core::decode_events().tag(1u << 1);
            runtime_.apply_pending(true);
        }
        // the elastic KV: the next window's cells backed before it runs (between windows, rare)
        if (runtime_.kv_floor > 0 && p + T + 64 > basalt::core::session_kv_cells(*runtime_.rsp)) {
            if (adapt_thr.joinable()) adapt_thr.join();
            if (copy_live) { runtime_.src.commit_exchanges(); copy_live = false; }   // as at the request's end
            if (!runtime_.kv_fit(*runtime_.rsp, p + runtime_.kKvGrowCells, runtime_.err, true, p + runtime_.kKvWindowCells)) {
                if (runtime_.kv_short) {
                    kv_stop = runtime_.err;
                    return 1;
                }
                return 2;
            }
        }
        basalt::core::decode_gaps().mark(18);
    return 0;
}

inline void QfnRequest::spawn_adapt(bool early) {
        
            if (runtime_.adapt_fast) {   // a round, or the next step of the one in flight, beside the draft
                if (!do_adapt && runtime_.xq_stage == 0) return;
                if (do_adapt) adapt_age = 0;
                basalt::core::decode_events().tag(1u << 0);
                basalt::core::decode_events().tag(1u << (6 + runtime_.xq_stage));
                adapt_early = early;
                adapt_thr.run([&, early] {
                    auto ms = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
                    const auto ta = Clock::now();
                    basalt::core::decode_events().stamp(8);
                    std::vector<QfnRuntime::Swap> ranked;
                    const bool pre = early && runtime_.xq_stage == 0;
                    if (pre) runtime_.rank_swaps(ranked, runtime_.xq_across);
                    const double ms_pre = ms(ta);
                    basalt::core::SpinWait spin;
                    while (early && adapt_go.load(std::memory_order_acquire) == 0) spin();
                    if (early && adapt_go.load(std::memory_order_acquire) != 1) return;
                    const auto tb = Clock::now();
                    adapt_ok = runtime_.xq_stage != 0 ? runtime_.xq_advance(false) : runtime_.xq_round(pre ? &ranked : nullptr);
                    basalt::core::decode_events().stamp(6);
                    std::string perr;
                    if (adapt_ok && !runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    basalt::core::decode_events().stamp(7);
                    basalt::core::decode_gaps().hms[20] += ms_pre + ms(tb);
                });
                return;
            }
            if (!do_adapt) return;
            adapt_age = 0;
            adapt_thr.run([&] {
                const auto ta = Clock::now();
                adapt_ok = runtime_.adapt();
                if (adapt_ok) {   // the evictions' residency, uploaded beside the draft
                    std::string perr;
                    if (!runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                }
                basalt::core::decode_gaps().hms[20] += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            });
        }

inline void QfnRequest::tier_before(int T, int e0, bool from_sfx) {
        // the window's adaptive step, decided before it runs (the window changes none of its inputs)
        adapt_ok = true;
        do_adapt = !runtime_.drive.d.usage.empty() && ((runtime_.rounds + 1) % runtime_.o.adapt_every) == 0 &&
                              (runtime_.adapt_fast ? runtime_.xq_stage == 0
                                          : (runtime_.pending.empty() && !copy_live && !runtime_.src.exchanges_staged()));
    adapt_early = false;
    adapt_go.store(0, std::memory_order_relaxed);   // 1 go, 2 dropped (the window failed)
    (void) T;
    (void) e0;
    (void) from_sfx;
        // the plain rounds copy into the slots at once, so they start after the window
        if (runtime_.adapt_fast) cv->set_served_hook([this] { spawn_adapt(true); });
}

inline void QfnRequest::tier_after_commit() {
        
            if (adapt_early) adapt_go.store(1, std::memory_order_release);
            else spawn_adapt(false);
            // the swaps' residency at the next window, then their RAM copies a share per draft
            applied_early = adapt_age == 1 && !do_adapt && !runtime_.pending.empty();
            if (applied_early || copy_live) {
                const bool boundary = applied_early, copies = applied_early || copy_live;
                if (copies) {
                    copy_live = true;
                    copy_done = false;
                }
                basalt::core::decode_events().tag(1u << 2);
                apply_thr.run([&, boundary, copies] {
                    if (boundary) {
                        runtime_.apply_pending(true, false);
                        std::string perr;
                        if (!runtime_.ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
                    }
                    if (copies) copy_done = !runtime_.src.exchanges_staged() || runtime_.src.copy_exchanges(8ull << 20);
                });
            }
        }

inline void QfnRequest::tier_finish() {
    int wait = 0;
    if (adapt_early) adapt_go.compare_exchange_strong(wait, 2);
    if (adapt_thr.joinable()) adapt_thr.join();
}

inline bool QfnRequest::run_window(int T, int e0) {
        // the prompt rows the first window folds in send no pick: the head scores them only roughly (exact logits
        // are kept where something reads them: a scored chunk, the logit-hash diagnostic, and a sampled window's
        // kept lists, which the drafter notes as candidates)
        if (first_window && e0 > 0 && ppl == nullptr && !runtime_.logit_hash && req_temperature <= 0.0f)
            cv->set_head_from(e0);
        return cv->run(T, window.data(), p, runtime_.win_pool_fn, runtime_.win_pool_user, outv.data(), runtime_.err);
}

inline void QfnRequest::relax_note_forced(int a) {
            if (relax_on) {
                if (outv[(size_t) a] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) a] == runtime_.o.relax_call_close) in_call = false;
            }
}

inline int QfnRequest::accept_drafts(int T) {
    int a = 0;
    if (relax_on) {
            // a draft the target's draw disagrees with is kept when the target gives it probability >= P,
            // outside tool calls and never for an added token; the row then emits the draft
            bool call = in_call;
            while (a < T - 1) {
                const int32_t d = window[(size_t) a + 1];
                if (d != outv[(size_t) a]) {
                    if (call || d >= runtime_.o.relax_special_from || cv->draft_prob(a) < req_relax) break;
                    outv[(size_t) a] = d;
                    ++relaxed_kept;
                }
                if (d == runtime_.o.relax_call_open) call = true;
                else if (d == runtime_.o.relax_call_close) call = false;
                ++a;
            }
            for (int i = 0; i <= a; ++i)
                if (outv[(size_t) i] == runtime_.o.relax_call_open) in_call = true;
                else if (outv[(size_t) i] == runtime_.o.relax_call_close) in_call = false;
    } else
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
    return a;
}

inline bool QfnRequest::after_accept(int T, int& a, int e0, bool from_sfx) {
        (void) e0;
        if (runtime_.logit_hash)   // the accepted rows' logits, per position (equality of the batch rows with these)
            for (int t = 0; t <= a; ++t) basalt::log::raw("LH -1 %lld %016llx\n", (long long) (p + t), (unsigned long long) runtime_.row_hash(t));
        if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
        if (tk_log != nullptr && tk_valid && !from_sfx && ppl == nullptr && req_temperature > 0.0) {
            std::vector<float> lg((size_t) runtime_.n_vocab);
            std::fprintf(tk_log, "W %d %d\n", T, a);
            for (int j = 0; j < T; ++j) {
                if (j > a) {   // past the first miss: only the draft side
                    std::fprintf(tk_log, "R %d - - T D", j);
                    for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                    std::fputc('\n', tk_log);
                    continue;
                }
                if (!cv->copy_logits(j, lg.data())) break;
                // top_k, top_p, then temperature over the survivors: the sampler chain without penalties
                std::vector<int> ord((size_t) runtime_.n_vocab);
                for (int i = 0; i < runtime_.n_vocab; ++i) ord[(size_t) i] = i;
                const int K = basalt::kernels::sampler_list_len(req_top_k, (int) runtime_.n_vocab);
                std::partial_sort(ord.begin(), ord.begin() + K, ord.end(), [&](int x, int y) {
                    return lg[(size_t) x] > lg[(size_t) y] || (lg[(size_t) x] == lg[(size_t) y] && x < y);
                });
                const double mx = lg[(size_t) ord[0]];
                std::vector<double> e((size_t) K);
                double s1 = 0.0;
                for (int i = 0; i < K; ++i) s1 += (e[(size_t) i] = std::exp((double) lg[(size_t) ord[(size_t) i]] - mx));
                int keep = K;
                double cum = 0.0;
                for (int i = 0; i < K; ++i) { cum += e[(size_t) i] / s1; if (cum >= (double) req_top_p) { keep = i + 1; break; } }
                double s2 = 0.0;
                for (int i = 0; i < keep; ++i) s2 += (e[(size_t) i] = std::exp(((double) lg[(size_t) ord[(size_t) i]] - mx) / (double) req_temperature));
                std::fprintf(tk_log, "R %d %d %d", j, j + 1 < T ? (int) window[(size_t) j + 1] : -1, (int) outv[(size_t) j]);
                std::fprintf(tk_log, " T");
                for (int i = 0; i < std::min(keep, 8); ++i) std::fprintf(tk_log, " %d:%.5f", ord[(size_t) i], e[(size_t) i] / s2);
                std::fprintf(tk_log, " D");
                for (int i = 0; i < tk_prev[j].n; ++i) std::fprintf(tk_log, " %d:%.5f", (int) tk_prev[j].id[i], (double) tk_prev[j].p[i]);
                std::fputc('\n', tk_log);
            }
            std::fflush(tk_log);
        }
        tk_valid = false;
    return true;
}

inline void QfnRequest::note_draft_probs(bool from_sfx) {
    wprob_n = std::min(runtime_.S_mtp, 16);
    if (draft_log != nullptr)
        for (int i = 0; i < wprob_n; ++i) wprob[i] = first_window || from_sfx ? -1.0f : dprob[(size_t) i];
}

template <class After>
inline bool QfnRequest::draft_next(int T, int a, int e0, bool from_sfx, bool will_draft, After& after_launch) {
    basalt::core::decode_gaps().mark(5);
    if (!from_sfx && e0 == 0) cum_adapt.observe(dprob.data(), T - 1, a);
    runtime_.mtp.set_cum_th(cum_adapt.th());
    bool drafted = true;
    draft_failed = false;
        if (will_draft && runtime_.mtp_on) {
            runtime_.mtp.set_launch_hook(after_launch);
            runtime_.mtp.set_cum_th(cum_adapt.th());
            runtime_.mtp.set_host_th(cum_adapt.th());
            if (const int32_t* top = cv->top_ids())
                runtime_.mtp.note_candidates(top, (a + 1) * basalt::core::Verifier::kTopIds);
            drafted = runtime_.mtp.draft_from(cv->final_R_all(), T, outv.data(), p, a, drafts.data(), runtime_.err, dprob.data(), &n_chain);
            runtime_.mtp.set_launch_hook(nullptr);
            if (tk_log != nullptr && drafted) {
                for (int j = 0; j < 16 && j < runtime_.S_mtp; ++j) {
                    tk_prev[j].n = runtime_.mtp.diag_top(j, 4, tk_prev[j].id, tk_prev[j].p);
                }
                tk_valid = true;
            }
        }
    return drafted;
}

inline void QfnRequest::window_timing(int T, int a, bool from_sfx, Clock::time_point round0, Clock::time_point tw0,
                                      Clock::time_point tw1, Clock::time_point tw2) {
    basalt::core::decode_events().stamp(0);
        {
            const Clock::time_point tw3 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
            ++dec_windows; dec_T += T;
            if (draft_log != nullptr) {
                std::fprintf(draft_log, "W %d %d %d %.3f %.3f %.3f", T, a, from_sfx ? 1 : 0, msd(round0, tw0),
                             msd(tw0, tw1), msd(tw1, tw3));
                for (int i = 0; i < wprob_n; ++i) std::fprintf(draft_log, " %.4f", wprob[i]);
                std::fputc('\n', draft_log);
            }
        }
}

void QfnRequest::decode_end() {
    if (apply_thr.joinable()) apply_thr.join();
    if (copy_live) runtime_.src.commit_exchanges();   // swaps applied early: the rest of their copies
    copy_live = false;
}

Flow QfnRequest::decode() { return decode_loop(*this); }

}  // namespace basalt::models::qwen4exp::serve
