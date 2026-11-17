#include "basalt/models/qwen4exp/shared_names.hpp"
#include "basalt/core/checkpoint_transfer.hpp"
#include "context.hpp"

namespace basalt::models::qwen4exp::generate {

int resident_plan(Context& context) {
    auto& o = *context.o;
    auto& g = *context.g;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& ss = *context.ss;
    auto& srcp = *context.srcp;
    auto& pool_store = *context.pool_store;
    auto& auto_resident_bytes = *context.auto_resident_bytes;
    auto& xcache = *context.xcache;
    auto& kv_tail_slots = *context.kv_tail_slots;
    auto& cache_fill_bytes = *context.cache_fill_bytes;
    auto& drive = *context.drive;
    auto& d_res = *context.d_res;

    // ================================ THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below)
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk (`PL`, the same fields, inside a chunk: the position is an estimate), and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's lend cap); measured again once the RAM
    // complement is pinned, since a source pins nothing before that
    auto measure_pinned_share = [&]() {
        if (srcp == nullptr || o.prefill_chunk <= 0) return;
        uint64_t pinned = 0, total = 0;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        basalt::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    };
    measure_pinned_share();
    // The share of a cache a lent prompt chunk may take: 90% when the copies are DMA from pinned RAM, 85% when host
    // copies are the limit (lending more only streams more through them).
    auto lend_pct_for_share = [] { return basalt::prefill::Prefill::pinned_share() >= 0.9 ? (int64_t) 90 : (int64_t) 85; };
    int64_t auto_lend_pct = lend_pct_for_share();
    // The largest chunk --prefill auto may pick: the operator's ceiling (--prefill auto:N) and never past the context,
    // but never below 8192.  A bare `auto` is cut back once the RAM complement is built (below) to what the lend can
    // fund, and again to the fewest-equal-chunks cap when a prompt is read; without the resident RAM mode there is no
    // complement to size it from, so it stays at 8192.
    const int64_t kMaxAutoChunk = basalt::kernels::cpu::expert_layout().max_blob > (3u << 19) ? 32768 : 16384;
    int64_t auto_ceiling = std::max<int64_t>(8192, std::min<int64_t>(o.prefill_auto_max, o.max_context));
    if (o.prefill_auto_fit && !o.resident_cpu_experts) auto_ceiling = 8192;
    // what experts left in the file read through the OS file cache: every expert of the largest layer, and one prompt
    // chunk's reads for the lent slots the copy leaves out (sized below, with the lent slots)
    basalt::core::FileCache file_cache;
    for (int64_t l = 0; l < g.n_layers; ++l)
        file_cache.layer_bytes = std::max<uint64_t>(file_cache.layer_bytes,
                                                    (uint64_t) g.n_expert * basalt::kernels::cpu::expert_layout().blob_bytes(l));
    // --serve: free places in the RAM copy take the evicted experts (no exchange buffers)
    const bool adapt_fast_on = o.serve && o.adapt_every > 0 && o.adapt_swaps > 0;
    // ---- the host buffers serving takes at its peak, before the RAM copy reads the free memory: RAM held for the prompt
    // path's two PLE row buffers at the largest chunk a prompt may be read in, and
    // the checkpoint blocks of every slot holding all the checkpoints it keeps
    {
        const int64_t chunk_max = o.prefill_auto ? auto_ceiling : o.prefill_chunk;
        bool ok = true;
        if (o.serve && chunk_max > 0) {
            for (int b = 0; ok && b < 2; ++b) ok = basalt::core::host_plan((size_t) chunk_max * (size_t) g.n_embd * sizeof(float));
        }
        if (ok && o.serve && o.prompt_cache > 0) {
            const size_t keep = (size_t) (o.prompt_cache + 1) * (size_t) std::max(o.batch, 1);
            auto reserve = [&](const basalt::core::SessionState& u) {
                basalt::core::ConversationStateSizes z;
                std::string ze;
                return !basalt::core::conversation_session_sizes(g, u, z, ze) ||
                       basalt::core::CheckpointBytes::allocator_type::reserve(z.gdn, keep, &basalt::core::kCheckpointPin);
            };
            ok = reserve(ss);
        }
        if (!ok) return fatal_start_error("out of host memory for the prompt path's and the checkpoints' buffers");
    }
    // the GPU cache's last slots are lent to the prompt path (and the elastic KV): their experts are kept in RAM too
    int64_t lend_from = -1;
    if (o.prefill_chunk > 0 && d_res != nullptr && xcache.slots() > 0) {
        int64_t chunk = o.prefill_chunk;
        const int64_t k = serve_loans::plan_lend(chunk, xcache, g, ss, o.prefill_auto, auto_ceiling,
                                                (int) auto_lend_pct);
        if (k > 0) lend_from = xcache.slots() - k;
    }
    // the elastic KV's tail keeps its experts in RAM too (a slot lent to the KV becomes a CPU miss read from RAM),
    // below it the prompt path's lend region as before: the RAM decides how far down that goes, as it does today
    if (kv_tail_slots > 0)
        lend_from = lend_from >= 0 ? std::max<int64_t>(0, lend_from - kv_tail_slots) : xcache.slots() - kv_tail_slots;
    // ---- the start-up picks for the settings the user left out (resident RAM, pool affinity and
    // size): one pure rule over what this machine and pack have, so a given setting is never replaced.  The pool is
    // built here, with the picks.
    {
        namespace bp = basalt::program;
        namespace cpu = basalt::kernels::cpu;
        const auto topo = basalt::platform::detect_cpu_topology(true, basalt::platform::CpuAffinity::All);
        bp::AutoInputs in;
        in.ram_total = basalt::platform::total_physical_memory();
        // the container-aware reading (a memory cap below the machine's RAM), the one the RAM copy is checked against
        in.ram_available = basalt::platform::available_memory();
        if (basalt::platform::HostMemory hm; basalt::platform::host_available_memory(hm) && hm.available > 0)
            in.ram_available = in.ram_available > 0 ? std::min(in.ram_available, hm.available) : hm.available;
        in.worker_cores = (int) topo.worker_cores.size();
        in.p_cores = topo.p_cores;
        in.e_cores = topo.e_cores;
        in.hybrid = topo.is_hybrid;
        in.cpu_quota = cpu::cgroup_cpu_quota();
        const uint64_t lent_bytes =
            lend_from >= 0 ? (uint64_t) (xcache.slots() - lend_from) * basalt::kernels::cpu::expert_layout().max_blob : 0;
        file_cache.prompt_file_bytes = std::min<uint64_t>(lent_bytes, basalt::platform::kPromptFileReadBytes);
        in.file_cache = file_cache;
        in.complement_full = expert_total_bytes - std::min(expert_total_bytes, cache_fill_bytes);
        // the RAM copy also holds the free places the adaptive swaps use (see set_spares)
        const uint64_t blob = basalt::kernels::cpu::expert_layout().max_blob;
        const uint64_t spares_full = adapt_fast_on ? (uint64_t) o.adapt_swaps * kSatBoost * blob : 0;
        // the lent slots' copies and the spares are kept as far as RAM allows: they do not decide whether the complement fits
        in.extra_full = lent_bytes + spares_full;
        in.floor_bytes = in.ram_available - basalt::core::ram_tier_room(
            in.ram_available,
            std::max(o.resident_headroom, basalt::core::plan_reserve(basalt::platform::ram_reserve())) +
                basalt::core::pin_tracking_bytes(basalt::core::host_plan_bytes()),
            o.resident_pin);
        bp::AutoGiven& gv = in.given;
        if (!o.resident_cpu_experts || o.resident_budget_given) {
            gv.resident = true;
            gv.resident_gb = (double) o.resident_budget / 1073741824.0;
        }
        if (o.pool_affinity_given) {
            gv.pool_affinity = true;
            gv.pool_affinity_value = o.pool_affinity == cpu::PoolAffinity::All    ? bp::AutoAffinity::All
                                     : o.pool_affinity == cpu::PoolAffinity::PCores ? bp::AutoAffinity::PCores
                                                                                    : bp::AutoAffinity::Auto;
        }
        if (o.pool_workers_given && o.pool_workers > 0) {
            gv.pool_workers = true;
            gv.pool_workers_value = o.pool_workers;
        }
        const bp::AutoPlan plan = bp::choose_auto(in);
        auto_resident_bytes = plan.resident_bytes;
        const cpu::PoolAffinity affinity = plan.pool_affinity == bp::AutoAffinity::All    ? cpu::PoolAffinity::All
                                           : plan.pool_affinity == bp::AutoAffinity::PCores ? cpu::PoolAffinity::PCores
                                                                                            : cpu::PoolAffinity::Auto;
        const std::string line = bp::auto_flags_line(plan);
        if (!line.empty()) {
            // the server reads the engine's status lines on stdout
            if (o.serve) { std::printf("%s\n", line.c_str()); std::fflush(stdout); }
            else basalt::log::info("basalt generate", "%s", line.c_str());
        }
        auto& pool = pool_store.emplace(plan.pool_workers, affinity);
        if (pool.is_hybrid() && pool.affinity() != cpu::PoolAffinity::All) {
            const char* aff_str = pool.affinity() == cpu::PoolAffinity::PCores ? "p-cores" : "auto";
            basalt::log::info("basalt generate", "hybrid CPU detected (%d P-cores / %d threads, %d E-cores), pool workers: %d, affinity: %s",
                         pool.p_cores(), pool.p_threads(), pool.e_cores(), pool.workers(), aff_str);
        }
        char quota_note[64] = "";
        if (plan.pool_workers_src == bp::AutoSource::Auto && in.cpu_quota > 0 &&
            plan.pool_workers == (std::max)(1, (int) in.cpu_quota - 1))
            std::snprintf(quota_note, sizeof quota_note, " (quota %.2f CPUs)", in.cpu_quota);
        basalt::log::info("basalt generate", "%d expert-pool workers%s + the host thread%s", pool.workers(), quota_note,
                     o.no_pool ? " (UNUSED: --no-pool)" : "");
        drive.d.pool = &pool;
    }
    auto& pool = *pool_store;

    context.measure_pinned_share = measure_pinned_share;
    context.lend_pct_for_share = lend_pct_for_share;
    context.auto_lend_pct = &auto_lend_pct;
    context.kMaxAutoChunk = &kMaxAutoChunk;
    context.auto_ceiling = &auto_ceiling;
    context.adapt_fast_on = &adapt_fast_on;
    context.file_cache = &file_cache;
    context.lend_from = &lend_from;
    context.pool = &pool;
    return resident_load(context);
}

}
