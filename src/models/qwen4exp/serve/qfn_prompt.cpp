// src/models/qwen4exp/serve/qfn_prompt.cpp - how Flash-Next reads a prompt's parts: sampling setup, windows, loans, the split.
#include "halo/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"

namespace halo::models::qwen4exp::serve {

bool QfnRequest::prompt_begin() {
    ov_copied = false;

    batch_fatal = false;
    il_parts = 0;   // the slots' windows run between this prompt's chunks
    il_ms = 0;
    runtime_.apply_pending(true);
    return runtime_.xq_drain();
}

Flow QfnRequest::set_sampling() {
    if (!runtime_.ver.set_logit_bias(req_bias, runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::next;
    }
    runtime_.ver.set_sampling(req_sp);
    // a list longer than the top_k kernels keep is sampled through sorted rows: their buffers before the first window
    if (!req_sp.greedy &&
        halo::kernels::sampler_list_len(req_sp.top_k, (int) runtime_.n_vocab) > halo::kernels::kSamplerNarrowMax &&
        !runtime_.ver.reserve_wide_sampler(runtime_.err)) {
        std::printf("ERR %s\n", runtime_.err.c_str());
        return Flow::next;
    }
    // relaxed acceptance: sampled text requests without a logit_bias; never the quality gate
    relax_on = req_relax > 0.0f && !req_sp.greedy && !req_bias_present && ppl == nullptr && gguard == nullptr;
    runtime_.ver.set_draft_probe(relax_on);
    in_call = false;   // inside a tool call (its tags seen last in the prompt or the output): exact
    for (size_t i = ids.size(); i-- > 0;)
        if (ids[i] == runtime_.o.relax_call_open || ids[i] == runtime_.o.relax_call_close) { in_call = ids[i] == runtime_.o.relax_call_open; break; }
    relaxed_kept = 0;
    runtime_.drive.d.pcie_num = std::max(0, std::min(256, (int) (runtime_.o.pcie_frac * 256.0 + 0.5)));
    hist_n = req_sp.penalty_last_n;
    runtime_.ver.set_history(hist_n > 0 ? runtime_.d_hist : nullptr, hist_n);
    return Flow::proceed;
}

void QfnRequest::limit_parts() {
    // A short read into a slot beside decoding slots is not split: each part pays the lend, refill and
    // side-thread costs again, and a checkpoint stalls the windows.  One that runs alone stops at its root,
    // and at its turn boundary only past --prompt-cache-root tokens since the root: a short history is
    // cheaper to read again than the extra part costs.  The rest of a read that gave way keeps its parts.
    if (in_slot && runtime_.batch_on() && resumed_chunk <= 0 && n - 1 - read_from <= std::max<int64_t>(runtime_.sp.chunk(), 1)) {
        if (n - 1 - read_from > runtime_.kReadAlone) {
            turn_at = -1;
            root_at = -1;
            media_at = -1;
        } else {
            if (turn_at - std::max(read_from, root_at) < runtime_.o.prompt_cache_root) turn_at = -1;
            if (media_at - std::max(read_from, root_at) < runtime_.o.prompt_cache_root) media_at = -1;
        }
    }
    // the parts after the first read their n-gram rows from memory: once the first part has its own, the rest
    // are read into the row cache beside it
    {
        int64_t first_to = 0;
        for (const int64_t to : {root_at, media_at, turn_at, n - 1})
            if (to > read_from) { first_to = to; break; }
        runtime_.sp.set_ple_warm(ids.data(), read_from, first_to, n - 1);
    }
}

Flow QfnRequest::read_to(int64_t to) {
    const auto source_at = [&](int64_t t) {
        return t == root_at ? ConvCheckpoint::kRoot : ConvCheckpoint::kTurnEnd;
    };
    runtime_.err.clear();
    const bool win = windows_ok(at, to);
    int64_t rto = to;
    if (win && to == n - 1 && !part_checkpoint(to) && ppl == nullptr && gguard == nullptr) {
        const int64_t p0 = at + (int64_t) runtime_.S * ((to - at) / runtime_.S);
        if (!runtime_.mtp_on || n - p0 <= runtime_.S_mtp) { rto = p0; fold_p0 = p0; }
    }
    if (win && !refill(runtime_.err)) {
        std::printf("ERR refilling a lent slot failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if (!win && !lend(to - at, runtime_.err)) {
        std::printf("ERR lending the prompt path its slots failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    // beside slots: a window between the serial steps of a short read keeps them from stopping
    auto ov_window = [&]() -> bool {
        if (!runtime_.batch_on()) return true;
        if (!runtime_.batch_step()) { batch_fatal = true; runtime_.err = "a batch window failed"; return false; }
        if (runtime_.mtp_on && !runtime_.mtp.use_main(runtime_.err)) return false;
        return true;
    };
    if (win && !ov_window()) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
    runtime_.ov_post_done = false;
    runtime_.ov_post = [&, to]() -> bool {
        if (part_checkpoint(to) && !runtime_.checkpoint_at(to, false, source_at(to))) return false;
        // the lent slots get their experts back here, beside the windows; the table follows on the main thread
        for (PfPart& p : runtime_.pf_parts) {
            if (p.lent.empty()) continue;
            std::string re;
            const halo::core::OnDevice on(p.dev);
            if (!refill_issue(p, re, false) || !p.cache->sync_queued(re)) { runtime_.ckpt_why = ": " + re; return false; }
        }
        ov_copied = true;
        return true;
    };
    runtime_.pp_part_ckpt = part_checkpoint(to) ? to : -1;
    const bool sp_ok = win ? (rto <= at || read_windows(at, rto, runtime_.err)) : read_part(at, to, runtime_.err);
    runtime_.pp_part_ckpt = -1;
    if (sp_ok && win && !ov_window()) { std::printf("ERR %s\n", runtime_.err.c_str()); return Flow::error; }
    if (!sp_ok) {
        if (yielded_at < 0 && (batch_fatal || !(runtime_.stop_req.load() || runtime_.stdin_lost.load() || runtime_.ram_stop.load()))) {
            halo::log::error("halo serve", "%s", runtime_.err.c_str());
            std::printf("ERR %s\n", runtime_.err.c_str());
            // a CUDA fault (an illegal address) poisons the context for the whole process, and unwinding
            // the destructors on it could hang until the watchdog: leave at once
            if (cudaPeekAtLastError() != cudaSuccess) {
                std::fflush(stdout);
                std::_Exit(1);
            }
            return Flow::error;
        }
        cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
        return Flow::quit;
    }
    at = to;
    resumed_chunk = 0;   // the later parts are planned as the read they continue planned them
    runtime_.ov_post = nullptr;
    if (part_checkpoint(to) && !runtime_.ov_post_done && !runtime_.checkpoint_at(to, !in_slot && !runtime_.batch_on(), source_at(to))) {
        std::printf("ERR saving a conversation checkpoint failed%s\n", runtime_.ckpt_why.c_str());
        return Flow::error;
    }
    return Flow::proceed;
}

Flow QfnRequest::prompt_end() {
    if (!refill(runtime_.err)) {
        std::printf("ERR refilling a lent slot failed: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    if (il_parts > 0)
        halo::log::info("halo batch", "the prompt was read in %lld parts, the slots decoding %.0f ms between "
                             "them", (long long) il_parts + 1, il_ms);
    return Flow::proceed;
}

}  // namespace halo::models::qwen4exp::serve
