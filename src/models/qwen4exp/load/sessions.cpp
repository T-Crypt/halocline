#include "basalt/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

int sessions(Context& context) {
    auto& o = *context.o;
    auto& srcs = *context.srcs;
    auto& model_file = *context.model_file;
    auto& err = *context.err;
    auto& g = *context.g;
    auto& K = *context.K;
    auto& wt = *context.wt;
    auto& ss = *context.ss;
    auto& sbuf = *context.sbuf;
    auto& main_cs = *context.main_cs;
    auto& ple_table = *context.ple_table;


    uint64_t solo_kv_bytes = 0;   // the solo session's state on CUDA0, reported once it is known to be a session of its own
    // ---- THE ELASTIC KV.  A server on one GPU backs only the first kKvElasticFloor cells of each session's K/V pools at
    // start; the rest of their VRAM goes to the expert cache as its tail, and a request that needs more cells takes
    // them back from that tail (kv_fit below), so a short request decodes with the larger cache.  Not with KV
    // streaming (its pools are a fixed set of VRAM pages) nor with --vram-elastic (that resizes the same tail).
    // With --batch a session's base is smaller: the slots' sessions are many, and an idle or short one would hold its
    // whole base out of the expert cache.
    constexpr int64_t kKvElasticFloor = 32768, kKvElasticFloorBatch = 4096;
    const int64_t kv_base = o.batch > 0 ? kKvElasticFloorBatch : kKvElasticFloor;
    const int64_t kv_floor = o.serve && o.kv_resident == 0 && !o.vram_elastic &&
                             o.max_context > kv_base ? kv_base : 0;
    basalt::core::qsa_set_kv_elastic(kv_floor);
    // ---- the session
    {
        const basalt::core::OnDevice on0(0);
        const int64_t hi0 = -1;
        if (cudaMalloc(&sbuf, basalt::core::session_bytes(g, o.max_context, K, 0, hi0)) != cudaSuccess) {
            basalt::log::error("basalt generate", "session state allocation failed");
            return 1;
        }
        solo_kv_bytes = basalt::core::session_bytes(g, o.max_context, K, 0, hi0);
        if (basalt::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            basalt::log::error("basalt generate", "session_init failed");
            return 1;
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            basalt::log::info("basalt generate", "KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) basalt::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt path or
        // the verifier reads it (--serve zeroes it per request when nothing is reused)
        basalt::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            basalt::log::error("basalt generate", "zeroing the session state failed");
            return 1;
        }
        if (!ss.ple.ready()) {
            basalt::log::error("basalt generate", "the PLE run is not ready after construction");
            return 1;
        }
        basalt::log::info("basalt generate", "PLE on, table %llu rows of %s",
                     (unsigned long long) ple_table.rows(), o.model.c_str());
    }

    // --batch: o.batch more sessions like the solo one (same context), one per slot of the batch windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.
    // Recommend, never force: a count the engine cannot run is said and adjusted (or batching left off) - the
    // engine still starts and serves one request at a time.
    std::vector<std::vector<SlotSession>> bslot_ss;
    std::vector<int64_t> slot_cap;   // each batch slot's context (cells)
    uint64_t slot_kv_cuda0 = 0;      // the batch slots' sessions on CUDA0 beyond the solo one: VRAM the expert cache does not get
    bool slot0_is_solo = false;      // slot 0 is the solo session (no separate solo state)
    if (o.batch != 0) {
        const int cap = basalt::kernels::kVerifyMaxT;
        const char* off = !o.serve ? "it needs --serve" : o.batch < 2 ? "it needs 2 or more slots" : nullptr;
        if (off != nullptr) {
            basalt::log::warn("basalt generate", "--batch %d is off: %s", o.batch, off);
            o.batch = 0;
        } else if (o.batch > cap) {
            basalt::log::warn("basalt generate", "--batch %d: a batch window holds at most %d rows, so %d "
                                 "slots", o.batch, cap, cap);
            o.batch = cap;
        }
    }
    if (o.batch > 0) {
        bslot_ss.resize(1);
        slot_cap.assign((size_t) o.batch, o.max_context);
        slot0_is_solo = true;   // the solo session's state is slot 0's
        int fit = o.batch;   // the slots that could be carved
        for (size_t k = 0; k < bslot_ss.size(); ++k) {
            const int dev = 0;
            const int64_t lo = 0, hi = -1;
            const basalt::core::OnDevice on_k(dev);
            uint64_t total = 0;
            for (int b = 0; b < fit; ++b) {
                const uint64_t bytes = basalt::core::session_bytes(g, slot_cap[(size_t) b], K, lo, hi);
                if (k == 0 && b == 0 && slot0_is_solo) {
                    bslot_ss[0].push_back(SlotSession(&ss, SlotDelete{false}));
                    basalt::log::info("basalt generate", "--batch: slot 0 on CUDA%d is the solo session (%.2f GiB)", dev,
                                 (double) bytes / 1073741824.0);
                    continue;
                }
                SlotSession u(new basalt::core::SessionState());
                void* buf = nullptr;
                if (cudaMalloc(&buf, bytes) != cudaSuccess || basalt::core::session_init(g, slot_cap[(size_t) b], K, buf, *u, lo, hi) == 0) {
                    cudaGetLastError();
                    if (buf != nullptr) cudaFree(buf);
                    size_t fb = 0, tb = 0;
                    cudaMemGetInfo(&fb, &tb);
                    basalt::log::warn("basalt generate", "--batch %d: slot %d's session does not fit on CUDA%d "
                                         "(%.2f GiB, %.2f GiB free); %d slot%s", o.batch, b, dev,
                                 (double) bytes / 1073741824.0, (double) fb / 1073741824.0, b, b == 1 ? "" : "s");
                    fit = b;
                    break;
                }
                basalt::core::session_zero(*u, g, nullptr, nullptr);
                bslot_ss[k].push_back(std::move(u));
                total += bytes;
                basalt::log::info("basalt generate", "--batch: slot %d on CUDA%d holds %lld tokens (%.2f GiB)", b, dev,
                             (long long) slot_cap[(size_t) b], (double) bytes / 1073741824.0);
            }
            cudaDeviceSynchronize();
            size_t fb = 0, tb = 0;
            cudaMemGetInfo(&fb, &tb);
            basalt::log::info("basalt generate", "--batch: %d slot sessions on CUDA%d (%.2f GiB in all); %.2f GiB free",
                         (int) bslot_ss[k].size(), dev, (double) total / 1073741824.0, (double) fb / 1073741824.0);
        }
        for (auto& v : bslot_ss) if ((int) v.size() > fit) v.resize((size_t) fit);
        if (fit < 2) {
            basalt::log::warn("basalt generate", "--batch is off: not two slot sessions fit (a smaller "
                                 "--max-context, or KV streaming, --kv-resident, makes them smaller)");
            bslot_ss.clear();
            slot_cap.clear();
            o.batch = 0;
            slot0_is_solo = false;
        } else {
            o.batch = fit;
            for (size_t k = 0; k < bslot_ss.size(); ++k)   // every slot session allocated, solo state included
                for (int b = 0; b < o.batch; ++b)
                {
                    const uint64_t sb = basalt::core::session_bytes(g, slot_cap[(size_t) b], K);
                    const basalt::core::SessionState& u = *bslot_ss[k][(size_t) b];
                    basalt::core::LoadProgress::kv(
                        basalt::core::LoadProgress::cuda(0),
                        sb + basalt::core::session_kv_mapped_bytes(u), slot_cap[(size_t) b], "slot",
                        sb + basalt::core::session_kv_bytes_for(u, u.max_cells));
                }
            slot_cap.resize((size_t) fit);
            // the sessions' VRAM is the expert cache's: say what it costs
            uint64_t kept = 0;
            for (int b = slot0_is_solo ? 1 : 0; b < o.batch; ++b)
                kept += basalt::core::session_bytes(g, slot_cap[(size_t) b], K);
            slot_kv_cuda0 = kept;
            const uint64_t per_slot = basalt::core::session_bytes(g, slot_cap[0], K);
            basalt::log::info("basalt generate", "--batch %d: %.2f GiB of KV cache and state per slot (%lld tokens), "
                                 "%.2f GiB in all; the slots beyond the solo one take %.2f GiB of VRAM on CUDA0 that "
                                 "the expert cache would otherwise hold", o.batch, (double) per_slot / 1073741824.0,
                         (long long) slot_cap[0], (double) (per_slot * (uint64_t) o.batch) / 1073741824.0,
                         (double) kept / 1073741824.0);
        }
    }

    // the drafter's K/V is elastic like the sessions' (MtpDrafter::load keeps it whole with a K/V per batch slot)
    basalt::core::qsa_set_kv_elastic(kv_floor);
    if (!slot0_is_solo)
        basalt::core::LoadProgress::kv(basalt::core::LoadProgress::cuda(0),
                                       solo_kv_bytes + basalt::core::session_kv_mapped_bytes(ss), o.max_context, "",
                                       solo_kv_bytes + basalt::core::session_kv_bytes_for(ss, ss.max_cells));
    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts: the driver can
    // refuse the draft weights after mapping tens of GiB of host pages.
    basalt::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) o.mtp.clear();
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        if (o.batch > 0) mtp.set_stores(o.batch, slot_cap);   // a draft K/V per batch slot
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const basalt::core::ModelGeometry draft_geometry{};
        if (!o.mtp.empty() && !mtp.load(basalt::core::MtpFiles::from(srcs), draft_geometry, ss, o.spec, err, o.mtp_window)) { basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str()); return 1; }
        mtp.set_ple_session(&ss);
    }
    basalt::core::qsa_set_kv_elastic(0);   // every later state: whole
    const bool mtp_on = !o.mtp.empty();
    const basalt::core::QsaState* const mtp_kv = mtp_on ? &mtp.kv_state() : nullptr;   // the draft layer's K/V, if any
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  Loaded first, the cache is sized around the native
    // head, and the head is not refused after the expert arena has registered tens of GiB of host pages, like the
    // drafter above.
    const basalt::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { basalt::log::error("basalt generate", "output.weight is missing"); return 1; }
    const int64_t n_vocab = wo->ne1;
    // --grammar on: the token texts the grammar walks, read from the pack's tokenizer files
    std::optional<basalt::program::grammar::PackVocab> grammar_vocab;
    if (o.serve && o.grammar) {
        std::vector<uint8_t> vj, tj;
        std::string gerr;
        basalt::program::grammar::PackVocab pv;
        if (!basalt::core::read_range(model_file.range("tokenizer/vocab.json"), vj, gerr) ||
            !basalt::core::read_range(model_file.range("tokenizer/token_type.json"), tj, gerr))
            return fatal_start_error("--grammar on: the pack has no tokenizer files: " + gerr);
        std::vector<int32_t> ends;
        for (int64_t e : o.eos_ids) ends.push_back((int32_t) e);
        if (!basalt::program::grammar::vocab_from_pack(std::string_view((const char*) vj.data(), vj.size()),
                                                       std::string_view((const char*) tj.data(), tj.size()),
                                                       (size_t) n_vocab, ends, pv, gerr))
            return fatal_start_error("--grammar on: " + gerr);
        grammar_vocab = std::move(pv);
    }
    basalt::core::NativeHead native_head;
    {
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_shards, g.n_embd, n_vocab, err)) {
            basalt::log::error("basalt generate", "%s%s", err.c_str(), vram_free_note().c_str());
            return 1;
        }
        basalt::log::info("basalt generate", "native head: %s, %llu bytes, in %.1f s",
                     basalt::ggml_type_name((uint32_t) native_head.type()),
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        (void) cudaGetLastError();
        basalt::log::error("basalt generate", "the logits buffer failed%s", vram_free_note().c_str());
        return 1;
    }


    context.kv_floor = &kv_floor;
    context.bslot_ss = &bslot_ss;
    context.slot_cap = &slot_cap;
    context.slot_kv_cuda0 = &slot_kv_cuda0;
    context.slot0_is_solo = &slot0_is_solo;
    context.mtp = &mtp;
    context.mtp_on = &mtp_on;
    context.mtp_kv = &mtp_kv;
    context.n_vocab = &n_vocab;
    context.grammar_vocab = &grammar_vocab;
    context.native_head = &native_head;
    context.d_logits = &d_logits;
    return sources(context);
}

}
