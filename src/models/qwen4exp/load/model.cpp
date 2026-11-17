#include "halo/model/rope_flags.hpp"
#include "halo/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace halo::models::qwen4exp::generate {

int model(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& no_spec = *context.no_spec;
    auto& rope_cfg = *context.rope_cfg;
    auto& err = *context.err;

    halo::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first (precedence: halo/model/rope_flags.hpp).
    {
        halo::model::RopeKeys keys;   // the model file's rope keys, when it carries any
        {
            // the model file is the authority on its own MoE shape (a pruned variant has fewer experts);
            // everything else in the geometry is unchanged
            try {
                halo::GgufFile model_gguf(o.native_shards.front());   // the metadata shard
                if (const halo::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
                if (const halo::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
                keys = halo::model::rope_keys(model_gguf, "qwen4exp");
            } catch (const std::exception& e) {
                halo::log::error("halo generate", "reading the model's expert shape from %s: %s",
                             o.model.c_str(), e.what());
                return 1;
            }
        }
        if (!halo::model::resolve_rope_scaling(o.rope, keys, o.max_context, "halo generate", rope_cfg, err)) {
            halo::log::error("halo generate", "%s: %s", o.model.c_str(), err.c_str());
            return 2;
        }
        halo::kernels::rope_scaling_set(rope_cfg);
    }
    halo::core::NativeEmbed native_embed;
    {
        if ((o.spec < 2 && !no_spec) || (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            halo::log::error("halo generate", "%s runs its experts in verify windows: it needs --spec T (T >= 2; "
                                 "--serve also takes 0) and --prefill CHUNK", o.model.c_str());
            return 2;
        }
        const halo::core::ModelGeometry g0;
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.native_shards, g0.n_embd, 248320, err)) {
            return fatal_start_error(err);
        }
        halo::core::set_native_embed(&native_embed);
        halo::log::info("halo generate", "%s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s), its first %lld rows in VRAM (%.1f MiB)",
                     o.model.c_str(), (double) halo::kernels::cpu::expert_layout().max_blob / 1e6,
                     halo::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count(),
                     (long long) native_embed.hot_rows(), (double) native_embed.hot_bytes() / 1048576.0);
    }
    // Tensors served from the source tensors are not also loaded into the weight arena.
    std::set<std::string> skip;
    {
        if (!halo::core::NativeDense::served_names(o.native_shards, true, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("output.weight");
        // a PLE key the dense index holds in BF16 stays in the arena
        if (!halo::core::NativeDense::keep_unquantized_ple_key(pack_files, skip, err)) {
            return fatal_start_error(err);
        }
        skip.insert("token_embd.weight");
    }
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        // --expert-profile-save's file from an earlier run of this model: the start places what that run's adaptive
        // tier learned instead of the model's profile (the same format; another model's file is refused by its shape)
        bool learned = false;
        if (o.serve && !o.expert_profile_save.empty()) {
            uint64_t bytes = 0;
            std::string perr;
            if (halo::platform::file_size(o.expert_profile_save, bytes) && bytes > 0) {
                if (halo::core::read_expert_profile(halo::core::FileRange(o.expert_profile_save), g.n_layers,
                                                      g.n_expert, profile, pslots, perr)) {
                    learned = true;
                    o.expert_profile = o.expert_profile_save;
                } else {
                    halo::log::warn("halo generate", "%s not used (%s): the engine's profile instead",
                                      o.expert_profile_save.c_str(), perr.c_str());
                    profile.clear();
                }
            }
        }
        if (!learned && !halo::core::read_expert_profile(halo::core::FileRange(o.expert_profile), g.n_layers,
                                                           g.n_expert, profile, pslots, err)) {
            return fatal_start_error(err);
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the file).
        // `--expert-cache 0` means auto: the profile ranks every pair.
        if (o.expert_cache == 0) o.expert_cache = -1;
        halo::log::info("halo generate", "profile %s: %zu ranked pairs, built for %lld slots",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only CUDA0's pairs
    // in `profile` below).  Empty without --expert-profile-save.
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    if (!o.expert_profile_save.empty()) profile_loaded = profile;

    context.g = &g;
    context.K = &K;
    context.skip = &skip;
    context.profile = &profile;
    context.profile_loaded = &profile_loaded;
    return layer_plan(context);
}

}
