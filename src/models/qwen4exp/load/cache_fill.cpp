#include "halo/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace halo::models::qwen4exp::generate {

int cache_fill(Context& context) {
    auto& o = *context.o;
    auto& err = *context.err;
    auto& profile = *context.profile;
    auto& model_bytes = *context.model_bytes;
    auto& expert_total_bytes = *context.expert_total_bytes;
    auto& cuda0_fixed_bytes = *context.cuda0_fixed_bytes;
    auto& src = *context.src;
    auto& srcp = *context.srcp;
    auto& xcache = *context.xcache;

    // ---- fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    uint64_t cache_fill_bytes = 0;   // what the GPU cache took from the profile (counted by its load phase)
    // The model's bytes against what the tiers will hold, from the same quantities the loader uses: the primary card
    // keeps its weights and the cache just sized, and the RAM copy the room past the headroom.  With the file-backed modes the rest is read from the SSD (the default
    // arena mode was checked against the RAM above).
    if (model_bytes > 0 && o.mmap_experts) {
        const auto free_ram = halo::core::conversation_available_memory();
        if (free_ram) {
            uint64_t cache_bytes_now = 0;
            if (!profile.empty() && srcp != nullptr) {
                const int64_t n = xcache.per_layer_admission()
                                      ? (int64_t) profile.size()
                                      : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
                for (int64_t i = 0; i < n; ++i)
                    cache_bytes_now += halo::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            }
            cache_bytes_now = std::min(cache_bytes_now, expert_total_bytes);
            const uint64_t vram = cuda0_fixed_bytes + cache_bytes_now;
            uint64_t outside = expert_total_bytes - cache_bytes_now;   // the experts no cache on the primary holds
            uint64_t ram = halo::core::ram_tier_room(
                *free_ram, std::max(o.resident_headroom, halo::core::plan_reserve(halo::platform::ram_reserve())), o.resident_pin);
            if (o.resident_cpu_experts && (o.resident_budget > 0 || o.resident_soft)) {
                const uint64_t margin = 256ull << 20;
                ram = ram > margin ? ram - margin : 0;
                if (o.resident_budget > 0) ram = std::min(ram, o.resident_budget);
            }
            if (model_bytes > vram + ram) {
                ram = std::min(ram, outside);
                if (model_bytes > vram + ram)
                    halo::log::warn("halo", "model %.1f GiB, free VRAM + RAM %.1f GiB: the rest is read from "
                                         "the SSD while generating (slow)",
                                 (double) model_bytes / 1073741824.0,
                                 (double) (vram + ram) / 1073741824.0);
            }
        }
    }
    if (!profile.empty() && srcp != nullptr) {
        // per layer, a full layer skips only its own pairs - each layer takes its hottest experts until its range is
        // full
        const bool per_layer = xcache.per_layer_admission();
        const int64_t want = per_layer ? (int64_t) profile.size()
                                       : std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        uint64_t fill_total = 0;
        for (int64_t i = 0; i < want; ++i)
            fill_total += halo::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        cache_fill_bytes = fill_total;
        if (model_bytes > 0) {
            if (srcp == &src) {
                halo::core::LoadProgress::plan("cuda0", cuda0_fixed_bytes + fill_total);
                halo::core::LoadProgress::plan("ram", expert_total_bytes - fill_total);
            } else {
                halo::core::LoadProgress::plan("cuda0", fill_total, true);
            }
        }
        halo::core::LoadProgress fill_progress(halo::core::LoadProgress::cuda(0), fill_total, srcp != &src);
        // the slots are handed out in profile order, then the blobs are read and copied together
        std::vector<std::pair<int32_t, int32_t>> fill_pairs;
        std::vector<int32_t> fill_slots;
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == halo::core::kNotResident) {
                if (per_layer) continue;
                break;
            }
            fill_pairs.push_back(profile[(size_t) i]);
            fill_slots.push_back(slot);
            fill_bytes += halo::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
        }
        if (!halo::core::fill_cache_slots(*srcp, xcache, 0, fill_pairs.data(), fill_slots.data(),
                                            (int64_t) fill_pairs.size(),
                                            [&](uint64_t bytes) { fill_progress.add(bytes); }, err)) {
            halo::log::error("halo generate", "the profile fill failed: %s", err.c_str());
            return 1;
        }
        prefilled += (int64_t) fill_pairs.size();
        fill_progress.finish();
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one blob-sized D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) halo::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            return fatal_start_error(err);
        }
        halo::log::info("halo generate", "pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
    }


    context.cache_fill_bytes = &cache_fill_bytes;
    return token_path(context);
}

}
