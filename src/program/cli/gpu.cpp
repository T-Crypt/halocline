#include "gpu.hpp"

#include "basalt/build_archs.hpp"
#include "basalt/log.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cuda_runtime.h>

namespace basalt::program::cli {

std::optional<int> select_gpus(CommonOptions& o, std::vector<GpuCard>& chosen_cards) {
    namespace bp = basalt::program;
    bp::GpuSpec spec;
    std::string gerr;
    if (!bp::parse_gpu_spec(o.gpu, spec, gerr)) {
        if (o.gpu_plan) {
            bp::GpuChoice bad;
            bad.error = gerr;
            std::printf("%s\n", bp::gpu_plan_json({}, bad).c_str());
        } else {
            basalt::log::error("basalt", "%s", gerr.c_str());
        }
        return 2;
    }
    bool known = false;
    const std::vector<bp::GpuCard> visible = bp::visible_gpus(known);
    const bp::GpuChoice choice = bp::choose_gpus(spec, visible, known);
    if (o.gpu_plan) {
        std::string lossy;
        for (const auto& f : o.lossy) lossy += (lossy.empty() ? "" : ",") + f.flag;
        std::printf("%s\n", bp::gpu_plan_json(visible, choice, o.arch, o.quality_flags, o.quality_window, o.plan_spec, o.max_context,
                                             o.model_name, lossy).c_str());
        return choice.error.empty() ? 0 : 2;
    }
    if (!choice.error.empty()) { basalt::log::error("basalt", "%s", choice.error.c_str()); return 2; }
    // Halocline runs on exactly one card (operator, 2026-11-16: one GPU, high bandwidth, sm_89 only).
    if (choice.cards.size() != 1) {
        basalt::log::error("basalt", "gpu %s: Halocline runs on exactly one GPU; %zu were chosen",
                           o.gpu.empty() ? "0" : o.gpu.c_str(), choice.cards.size());
        return 2;
    }
    for (const std::string& s : choice.notes) basalt::log::info("basalt", "%s", s.c_str());
    if (!choice.warning.empty()) basalt::log::warn("basalt", "%s", choice.warning.c_str());
    bp::apply_gpus(visible, choice.cards);
    int seen = 0;
    if (cudaGetDeviceCount(&seen) != cudaSuccess || seen < (int) choice.cards.size()) {
        basalt::log::error("basalt", "gpu %s: CUDA sees %d of the %zu GPUs chosen", o.gpu.empty() ? "0" : o.gpu.c_str(),
                           seen, choice.cards.size());
        return 2;
    }
    // every chosen card needs device code of its own architecture in this build (build_archs.hpp)
    for (int d = 0; d < (int) choice.cards.size(); ++d) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, d) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;   // device_info's own check stops the start then
        }
        if (!basalt::build_has_arch(major, minor)) {
            const int i = choice.cards[(size_t) d];
            basalt::log::error("basalt", "gpu: %s is sm_%d%d and this build has device code for %s only; build with "
                                         "-DCMAKE_CUDA_ARCHITECTURES=\"120;89\" for Blackwell and Ada cards together",
                               bp::gpu_label(visible, i).c_str(), major, minor, basalt::build_archs());
            return 2;
        }
    }
    bp::GpuCard card = choice.cards[0] >= 0 && (size_t) choice.cards[0] < visible.size() ? visible[(size_t) choice.cards[0]]
                                                                                           : bp::GpuCard{};
    basalt::log::info("basalt", "gpu: %s%s", card.name.empty() ? "GPU" : card.name.c_str(),
                      card.smi >= 0 ? (" (nvidia-smi " + std::to_string(card.smi) + ")").c_str() : "");
    chosen_cards = {card};
    return std::nullopt;
}

}  // namespace basalt::program::cli
