// The whole start of a Flash-Next engine, in order: its stack frames hold the resources while the server runs.
#include "halo/core/shipped_profiles.hpp"
#include "halo/models/qwen4exp/shared_names.hpp"
#include "halo/platform/file.hpp"
#include "context.hpp"
#include "../../../program/generate/run.hpp"
#include "models/qwen4exp/adapter.hpp"

namespace halo::models::qwen4exp {

int run_engine(Options& o, std::vector<program::GpuCard>& chosen_cards) {
    using namespace halo::models::qwen4exp::generate;
    Context context;
    std::string& ppl_base = o.ppl_base;
    std::string& ppl_out = o.ppl_out;
    halo::core::ModelSources srcs;
    halo::core::ModelFile model_file;
    {
        std::string merr;
        if (!model_file.open(o.model, merr) || !model_file.load_meta(merr) || !model_file.register_gguf_view(merr)) {
            return fatal_start_error("--model: " + merr);
        }
        halo::core::model_sources(model_file, srcs);
        // the source tensors (embedding, PLE table, native projections, head) are served through the GGUF view of the file
        o.native_shards = {o.model};
        if (srcs.dense.empty() || srcs.dense_index.empty() || srcs.experts.empty() || srcs.experts_layout.empty() ||
            srcs.hc8.empty()) {
            return fatal_start_error(o.model + " lacks the dense, dense.index, experts, experts.layout or hc8 section");
        }
        if (!srcs.mtp_dense.empty() && !srcs.mtp_dense_index.empty() && !srcs.mtp_experts.empty()) o.mtp = o.model;
        // the expert profile: the one --expert-profile names, else the engine's for this model when it has one
        uint64_t prof_bytes = 0;
        if (!o.expert_profile.empty()) {
            if (!halo::platform::file_size(o.expert_profile, prof_bytes) || prof_bytes == 0) {
                return fatal_start_error("--expert-profile " + o.expert_profile + ": no such file, or it is empty");
            }
        } else {
            const std::string key = cli::profile_key(model_file.meta());
            const std::string miss = halo::core::missing_shipped_profile(halo::platform::install_root() + "/data/profiles", cli::shipped_profiles(key));
            if (!miss.empty()) return fatal_start_error(miss);
            const std::string prof = halo::platform::install_root() + "/data/profiles/" + key + ".prof";
            if (!key.empty() && halo::platform::file_size(prof, prof_bytes) && prof_bytes > 0) o.expert_profile = prof;
            else {
                halo::log::info("halo generate", "no expert profile for %s", key.empty() ? "this model" : key.c_str());
                if (o.resident_cpu_experts) {
                    halo::log::error("halo generate", "expert profile not found: %s; ship data/profiles with the engine "
                                       "(or pass --mmap-experts without --resident-experts)",
                                       key.empty() ? "the model names no architecture or layer and expert counts" : prof.c_str());
                    return 2;
                }
            }
        }
        // the 8-bit hyper-connection copies
        std::map<std::string, halo::core::Hc8Copy> table;
        if (!halo::core::hc8_read(srcs.hc8, table, merr)) {
            return fatal_start_error(o.model + ": " + merr);
        }
        halo::core::hc8_set_table(std::move(table));
    }
    const halo::core::PackFiles pack_files = halo::core::PackFiles::from(srcs);
    const halo::core::ExpertFiles expert_files = halo::core::ExpertFiles::from(srcs);
    if (o.serve && o.conversation_cache_disk_mib > 0 && o.conversation_cache_dir.empty()) {
        halo::log::error("halo", "--conversation-cache-disk-mib needs --conversation-cache-dir");
        return 2;
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        halo::log::warn("halo serve", "conversation caching is disabled by %s",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    if (const auto code = cli::resolve_model_options(o)) return *code;
    if (o.serve) halo::platform::watch_resident_ram(o.resident_cpu_experts, 0);
    if (const auto code = cli::prepare_input_options(o)) return *code;
    halo::core::qsa_set_kv_int8(o.kv == "int8");
    halo::core::qsa_set_kv_q4(o.kv == "q4_0");   // 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    halo::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (const auto code = cli::validate_kv_streaming(o)) return *code;
    halo::core::qsa_set_kv_resident(o.kv_resident);
    const bool no_spec = cli::normalize_speculation_options(o);
    halo::core::layer_set_shared_early(true);
    {
        // the source tensors hold the PLE table
        try {
            const halo::GgufModel model(o.native_shards);
            if (model.find("per_layer_token_embd.weight") == nullptr) {
                halo::log::error("halo generate", "%s has no per_layer_token_embd.weight", o.model.c_str());
                return 2;
            }
        } catch (const std::exception& e) {
            halo::log::error("halo generate", "%s: %s", o.model.c_str(), e.what());
            return 2;
        }
    }
    halo::kernels::RopeScaling rope_cfg;
    if (const auto code = cli::validate_generation_options(o, vram_floor_mib())) return *code;

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI), so on a CPU without those features it
    // does not fail - it executes an illegal instruction at some unpredictable token.  Refusing at second zero is the
    // whole point of this check.
    // ... and nothing runs on a CPU without AVX2: every CPU expert kernel is AVX2 at least (the AVX-512 ones are
    // chosen above it), and so is ggml-cpu, which the native pack's layout load initializes next.  Refused here, by name, rather than an illegal instruction in the first expert.
    if (!halo::kernels::cpu::cpu_avx2_ok()) {
        halo::log::error("halo generate", "this CPU (%s) does not support AVX2 with FMA and F16C, which this engine's "
                             "CPU expert kernels need",
                     halo::kernels::cpu::cpu_name().c_str());
        return 2;
    }

    std::string err;
    // Where the experts live.  The model keeps every quantized tensor in its GGUF form, so its experts run in verify
    // windows only (--spec) and its dense projections, head and embedding come from the source tensors.
    {
        const halo::core::ModelGeometry g0;
        std::vector<uint8_t> text;
        bool layout_ok = halo::core::read_range(srcs.experts_layout, text, err) &&
                         halo::kernels::cpu::expert_layout_load_text(std::string(text.begin(), text.end()), g0.n_layers,
                                                                       g0.n_expert, err, g0.n_embd, g0.n_ff);
        if (!layout_ok) {
            return fatal_start_error(err);
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated: an unsupported type would otherwise fail from inside the first verify window or leave
        // the prompt path's fp16 buffer unwritten.
        const auto& lay = halo::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!halo::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s%s%s/%s (ggml types %d/%d/%d), which "
                                               "this engine has no GPU kernels for: format not supported for decode",
                              (long long) l, halo::ggml_type_name((uint32_t) f.gu_type),
                              f.up_type != f.gu_type ? "+" : "",
                              f.up_type != f.gu_type ? halo::ggml_type_name((uint32_t) f.up_type) : "",
                              halo::ggml_type_name((uint32_t) f.d_type), f.gu_type, f.up_type, f.d_type);
                return fatal_start_error(msg);
            }
            // Q6_K experts run on the fast kernels only (decode-once grouped / device-planned tail, MMQ prefill, CPU
            // dot): a pair or shape outside them is refused rather than routed to a general path.
            if ((f.gu_type == 14 || f.d_type == 14) &&
                (!halo::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type) ||
                 !halo::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) || !halo::prefill::mmq::fits(f.d_type, f.n_embd))) {
                char msg[224];
                std::snprintf(msg, sizeof msg, "layer %lld's experts are %s/%s: Q6_K experts have fast kernels only "
                                               "for the covered pairs and shapes",
                              (long long) l, halo::ggml_type_name((uint32_t) f.gu_type),
                              halo::ggml_type_name((uint32_t) f.d_type));
                return fatal_start_error(msg);
            }
        }
        // one line per run of layers with the same formats: the kernels each decode and prefill path uses
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size();) {
            const auto& f = lay.fmt[(size_t) l];
            int64_t e = l;
            while (e + 1 < (int64_t) lay.fmt.size() && lay.fmt[(size_t) e + 1].gu_type == f.gu_type &&
                   lay.fmt[(size_t) e + 1].up_type == f.up_type && lay.fmt[(size_t) e + 1].d_type == f.d_type)
                ++e;
            // a layer whose up rows have their own type shows both: gate+up/down
            const bool mixed = f.up_type != f.gu_type;
            halo::log::info("halo generate", "experts %s%s%s/%s, layers %lld-%lld: decode %s, prefill %s, CPU tier dot",
                         halo::ggml_type_name((uint32_t) f.gu_type), mixed ? "+" : "",
                         mixed ? halo::ggml_type_name((uint32_t) f.up_type) : "",
                         halo::ggml_type_name((uint32_t) f.d_type), (long long) l, (long long) e,
                         halo::kernels::dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)
                             ? "device-planned tail + grouped kernels" : "grouped kernels",
                         halo::prefill::mmq::fits(f.gu_type, 2 * f.n_ff) && halo::prefill::mmq::fits(f.up_type, 2 * f.n_ff) &&
                                 halo::prefill::mmq::fits(f.d_type, f.n_embd)
                             ? "MMQ" : "dequant GEMM");
            l = e + 1;
        }
    }
    // The PCIe share of the missed experts.  The GPU's SMs read that share over the link (the copy kernel), so on a
    // slower link it must shrink or the window waits for it.  The real H2D bandwidth is probed once; from 20 GB/s up
    // the default stays.
    if (o.pcie_frac < 0.0) {
        const double base = 0.55;
        std::string bursts;
        const double bw = probe_pcie_h2d_gbps(&bursts);
        if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            halo::log::info("halo generate", "PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)", bw, bursts.c_str(), o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            halo::log::warn("halo generate", "PCIe probe failed -> pcie_frac default %.2f", base);
        }
    }
    if (!halo::kernels::cpu::cpu_avx512_ok())
        halo::log::warn("halo generate", "this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)",
                     !halo::kernels::cpu::cpu_avx2_ok() ? "ggml-cpu vec_dot (no AVX2)" : "AVX-2");

    context.o = &o;
    context.ppl_base = &ppl_base;
    context.ppl_out = &ppl_out;
    context.chosen_cards = &chosen_cards;
    context.srcs = &srcs;
    context.model_file = &model_file;
    context.pack_files = &pack_files;
    context.expert_files = &expert_files;
    context.no_spec = &no_spec;
    context.rope_cfg = &rope_cfg;
    context.err = &err;
    return halo::models::qwen4exp::generate::model(context);
}

}  // namespace halo::models::qwen4exp
