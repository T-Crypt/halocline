// src/models/qwen4exp/serve/verify.cpp - verify operations for the persistent engine.
#include "halo/models/qwen4exp/shared_names.hpp"
#include "qfn_state.hpp"

namespace halo::models::qwen4exp::serve {


int QfnRuntime::init_verify() {

    d_hist = nullptr;
    hist_stage = std::vector<int32_t>(kHistSlots, -1);
    if (cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
        halo::log::error("halo serve", "the penalty-history allocation failed");
        return 1;
    }

    vh.d_res = thits.d_res;
    vh.h_res = host_res.empty() ? nullptr : host_res.data();
    vh.cache_base = thits.cache_base;
    vh.blob = thits.blob;
    vh.slot_off = xcache.slot_offsets();   // the device plan's pointers
    vh.n_slots = xcache.slots();

    ver.set_route_tail_skip(o.route_tail_skip);
    ver.set_route_prior(o.route_prior);
    ver.set_main_priority(o.batch > 0);   // beside a prompt read the slots' windows must win the SMs
    ver.set_cpu_warm(drive.d.pool, drive.d.src);
    if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max({o.spec, o.batch, 2}), err) ||
        (mtp_on && !mtp.bind(wt, &native_head, ver.final_R_all(), err))) {
        halo::log::error("halo serve", "%s", err.c_str());
        return 1;
    }
    // the verifier gets the slot sessions
    if (o.batch > 0) {
        std::vector<halo::core::SessionState*> ptrs;
        for (auto& u : bslot_ss[0]) ptrs.push_back(u.get());
        if (!ver.init_slots(ptrs, err)) {
            halo::log::error("halo serve", "--batch: %s", err.c_str());
            return 1;
        }
    }
    // the pool the verify windows call
    win_pool_fn = &drive_pool_multi;
    win_pool_user = (void*) &drive;
    return 0;
}

int QfnRuntime::capture_windows() {

    // every window width's graph is captured before READY, not by the first request that needs it
    if (o.serve && !ver.warm_graphs(err)) {
        halo::log::error("halo serve", "capturing the verify windows failed: %s", err.c_str());
        return 1;
    }
    // the solo windows of every slot session, so a lone request in any slot runs the solo path in place
    if (o.serve && slot0_is_solo && o.batch > 1) {
        const auto tw = Clock::now();
        size_t free0 = 0, total0 = 0;
        cudaMemGetInfo(&free0, &total0);
        for (int b = 1; b < o.batch; ++b) {
            make_view(rviews[(size_t) b], b);
            if (!ver.warm_session(&rviews[(size_t) b], err)) {
                halo::log::error("halo serve", "capturing slot %d's solo windows failed: %s", b, err.c_str());
                return 1;
            }
        }
        size_t free1 = 0;
        cudaMemGetInfo(&free1, &total0);
        halo::log::info("halo serve", "solo windows of %d more slot sessions captured in %.0f ms (%.1f MiB of VRAM)",
                     o.batch - 1, std::chrono::duration<double, std::milli>(Clock::now() - tw).count(),
                     (double) ((int64_t) free0 - (int64_t) free1) / 1048576.0);
    }
    S = o.spec;
    S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
    if (mtp_on && S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
    if (mtp_on && !mtp.prewarm(err)) {
        halo::log::error("halo serve", "capturing the draft graphs failed: %s", err.c_str());
        return 1;
    }
    ple_table.rearm_keepalive();   // a long load must not lapse the SSD keep-alive window before the first request
    // the tiers' placement at READY, which PROBE adapt puts back
    res_at_start = host_res;
    prof_rank_start = prof_rank;
    // the elastic KV's ceiling with everything loaded; the server's copy follows it from here
    kv_ceiling_note();
    if (kv_ceiling_sent >= 0) {
        const int n = std::max(o.batch, 1);
        halo::log::info("halo serve", "elastic KV: about %lld cells fit in the %d session%s (%lld at full "
                          "context) before the expert cache reaches the %lld slots a prompt chunk needs",
                          (long long) kv_ceiling_sent, n, n == 1 ? "" : "s", (long long) n * o.max_context,
                          (long long) cache_floor_slots());
    }
    std::printf("INFO %s\n", halo::program::spill_words(tier_spill).c_str());   // experts only the file holds
    std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
    std::fflush(stdout);

    rounds = 0;
    sfx.emplace(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
    policy.emplace(S);   // MTP or lookup window, learned over the whole process
    // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images and videos.  The
    // file is one or more halo-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt
    // order, one per picture and one per two-frame video group; each record's rows go to its run of <|image_pad|> or
    // <|video_pad|> tokens, whose M-RoPE positions are Qwen3-VL's (mrope_positions.hpp): t = p, h = p + y,
    // w = p + x, and the text after the run continues at p + max(nx, ny).
    mrope_identity = true;

    // ---- VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache until that
    // much VRAM is free for other programs, or grows it back towards its full size when more than that is free.
    // Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back the cache's
    // last segments: the experts in their slots become CPU misses, as any expert outside the cache is (no output
    // changes beyond what a smaller cache decodes; in the resident RAM mode each is first copied to a RAM home),
    // the prompt path's loan moves down to the end of what is left
    // (a smaller chunk when it no longer fits).  A grow maps them again and puts each slot's expert back (or, when
    // the adaptive tier already brought that one back, the most-routed missing expert of the same layer); the
    // adaptive tier carries on from there.
    chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache

    // ---- the elastic KV (kKvElasticFloor): a session's K/V pools back their cells on demand, out of the expert
    // cache's tail.  `kv_fit` backs `cells` of a session before anything writes them: the work in flight lands,
    // the cache gives back its last segments (their experts become CPU misses, read from their RAM copies), and the
    // pools map that VRAM, zeroed.  `kv_release` gives a session's cells past `cells` back and the cache maps its
    // tail again, each returned slot refilled from RAM.  Both run between windows, never inside one.
    // a request's room past its prompt, and each later step (batch slots: a smaller step, as their base)
    kKvGrowCells = o.batch > 0 ? 4096 : 16384;
    kv_moves = 0;

    kv_short = false;   // the last kv_fit refused: the cells would take the cache below its floor
    kv_keep_routed = false;
    return 0;
}

}  // namespace halo::models::qwen4exp::serve
