// src/models/qwen4exp/serve/qfn_commands.cpp - the protocol commands that reach Flash-Next's tiers.
#include "halo/program/qwen_forms.hpp"
#include "models/qwen4exp/adapter.hpp"
#include "halo/models/qwen4exp/shared_names.hpp"
#include "qfn_request.hpp"
#include "halo/vision/ffmpeg_api.hpp"
#include "../../../program/tokenizer/unicode_flags.hpp"
#include "../../../program/cli/tokens.hpp"

namespace halo::models::qwen4exp::serve {

bool QfnRuntime::next_batch_window() { return batch_step(); }

void QfnRuntime::script_command(const std::string& line) {
    char* e1 = nullptr;
    const long b = std::strtol(line.c_str() + 8, &e1, 10);
    char* e2 = nullptr;
    const unsigned long long seed = std::strtoull(e1, &e2, 10);
    std::vector<int64_t> sids;
    std::string se;
    if (b < 0 || b >= (long) bs.size() || e2 == e1 || !cli::parse_i64_list(e2, sids, se)) {
        std::printf("ERR BSCRIPT: bad slot or list\n");
        std::fflush(stdout);
        return;
    }
    bscript[(size_t) b].first.assign(sids.begin(), sids.end());
    bscript[(size_t) b].second = seed;
}

void QfnRuntime::probe_command(const std::string& line, std::string& perr) {
if (line.rfind("PROBE set ", 0) == 0) {   // PROBE set workers=K share=F: live values for the next requests
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            std::istringstream kvs(line.substr(10));
            for (std::string kv; kvs >> kv;) {
                const size_t eq = kv.find('=');
                const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
                const double v = eq == std::string::npos ? -1.0 : std::atof(kv.c_str() + eq + 1);
                if (key == "workers" && v >= 1) pool.set_active_workers((int) v);
                else if (key == "share" && v >= 0 && v <= 1) sp.set_cpu_share(v);
                else perr = "PROBE: bad setting " + kv;
            }
            if (perr.empty()) std::printf("PROBE end workers=%d\n", pool.active_workers());
        }
    } else if (line == "PROBE adapt" || line.rfind("PROBE adapt ", 0) == 0) {
        {
            if (ba_job) { ba_thr.join(); ba_job = false; }
            (void) probe_adapt(line, perr);
        }
    } else if (line != "PROBE rates") perr = "PROBE: expected PROBE rates, PROBE set or PROBE adapt";
    else {
        if (ba_job) { ba_thr.join(); ba_job = false; }
        halo::program::ProbeEnv penv;
        penv.pool = &pool;
        penv.ram_blob = [this](int64_t l, int64_t e) { return srcp->resident_blob(l, e); };
        if (const halo::kernels::cpu::ExpertLayout& lay = halo::kernels::cpu::expert_layout(); lay.native) penv.fmt = lay.fmt;
        penv.n_layers = g.n_layers;
        penv.n_expert = g.n_expert;
        halo::program::probe_rates(penv, stdout, perr);
    }
}

void QfnRuntime::sleep_command(bool down, std::string& verr) {
    // SLEEP drops the RAM copy's registration with the driver (a process checkpoint would map the copy into every
    // context again on restore); WAKE registers it again while the cache refills from the ranges already back.  A
    // registration that cannot come back fails WAKE: the server then starts the engine again.
    std::string reply, uerr;
    const auto tp = Clock::now();
    const bool repin = !down && src.complement_unpinned();
    if (repin) src.repin_start(4);
    vram_command(down ? "VRAM " + std::to_string(1ll << 30) : std::string("VRAM"), verr, true, &reply);
    if (repin) {
        if (!src.repin_finish(uerr) && verr.empty()) verr = "WAKE: " + uerr;
        halo::log::info("halo serve", "WAKE: the RAM copy registered again beside the caches' refill "
                          "in %.0f ms", std::chrono::duration<double, std::milli>(Clock::now() - tp).count());
    }
    if (down && verr.empty() && src.complement_ready()) {
        const auto tu = Clock::now();
        if (src.unpin_complement(uerr))
            halo::log::info("halo serve", "SLEEP: the RAM copy's registration dropped in %.0f ms",
                              std::chrono::duration<double, std::milli>(Clock::now() - tu).count());
        else
            halo::log::info("halo serve", "SLEEP: the RAM copy stays registered (%s)", uerr.c_str());
    }
    if (verr.empty()) std::fputs(reply.c_str(), stdout);
}

void QfnRuntime::between_requests(const std::string& line) {
    // every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
    if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
        Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
        save_profile("periodic");
    if (ba_job) { ba_thr.join(); ba_job = false; }   // (a request runs the same copy steps itself)
}

void QfnRuntime::after_input_start() { sp.should_stop = [&] { return halted(); }; }

void QfnRuntime::print_info() {
    // what the server's Monitor tab shows (the server skips unknown lines until READY)
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t slots_primary = (int64_t) xcache.slots();
        const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
        const int64_t slots_all = slots_primary, mib_all = mib_primary;
        // the pack's quantization label (the last word of its name, without the publisher's prefixes) and the PCI
        // bus id of the card this engine computes on: what a benchmark row records
        std::string quant_label, gpu_ids;
        {
            const halo::core::Json* nm = model_file.meta().get("name");
            quant_label = halo::program::pack_quant_label(nm != nullptr && nm->is_string() ? nm->s : std::string());
            char bus[32] = {};
            if (cudaDeviceGetPCIBusId(bus, sizeof bus, 0) == cudaSuccess) gpu_ids = bus;
        }
        std::printf("INFO logit_bias=%lld context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                    "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                    "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                    "conversation_cache_mib=%lld conversation_cache_slots=%d "
                    "conversation_cache_min_free_mib=%lld conversation_cache_min_tokens=%d tail_role_token=%lld vram_elastic=%d%s quant=%s gpus=%s engine=" HALO_VERSION "\n",
                    (long long) n_vocab, (long long) o.max_context, o.kv.c_str(),
                    (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                     ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                    (long long) slots_all, (long long) mib_all,
                    (long long) slots_primary, (long long) mib_primary,
                    o.spec, o.mtp_max_t,
                    o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                    (long long) ((o.mmap_experts ? src.resident_bytes() : halo::kernels::cpu::expert_layout().total) >> 20),
                    pool.workers(), o.pcie_frac,
                    (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                    (long long) o.conversation_cache_min_free_mib, o.conversation_cache_min_tokens, (long long) o.tail_role_token, o.vram_elastic && xcache.segmented() ? 1 : 0,
                    o.batch > 0 ? (" batch_slots=" + std::to_string(o.batch) +
                                   " slot_cache=" + std::to_string(o.prompt_cache > 0 ? 1 : 0) +
                                   " slot0_solo=" + std::to_string(slot0_is_solo ? 1 : 0) +
                                   " solo_any_slot=" + std::to_string(slot0_is_solo ? 1 : 0)).c_str() : "",
                    quant_label.c_str(), gpu_ids.c_str());
        // whether the in-engine encoder reads video (FFmpeg found at run time), for the server's start line
        if (o.vision_in_engine) std::printf("%s\n", halo::vision::video_line().c_str());
    }
}

Flow QfnRuntime::run_request() {
    QfnRequest request(*this);
    return request.execute();
}

int QfnRuntime::init_before_input() {
    if (const int code = init_prompt()) return code;
    if (const int code = init_verify()) return code;
    if (const int code = init_conversations()) return code;
    return init_tiers();
}

int QfnRuntime::init_after_input() {
    if (const int code = init_memory()) return code;
    if (const int code = capture_windows()) return code;
    return init_batch();
}

void QfnRuntime::at_exit() {
    for (const std::string& r : ver.route_report()) halo::log::info("halo serve", "%s", r.c_str());
    save_profile("exit");   // QUIT, or the server closed stdin
}

int run(Context& context) {
    QfnRuntime runtime(context);
    runtime.pre_ = {"qwen35", halo::program::tokenizer::pretok_qwen35};
    return runtime.execute();
}

}  // namespace halo::models::qwen4exp::serve
