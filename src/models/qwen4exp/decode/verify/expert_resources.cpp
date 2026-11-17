#include "common.hpp"
#include "expert_plan.hpp"
#include "halo/core/load_progress.hpp"
#include "halo/models/qwen4exp/decode/verify.hpp"
#include "halo/core/diag.hpp"
#include "halo/models/qwen4exp/decode/native_head.hpp"
#include "halo/core/on_device.hpp"
#include "halo/platform/compiler.hpp"
#include "halo/kernels/iq_kernels.hpp"
#include "halo/kernels/cpu/expert_layout.hpp"
#include "halo/models/qwen4exp/geometry.hpp"
#include "halo/kernels/cpu/pool.hpp"
#include "halo/kernels/pdl.hpp"
#include "halo/core/progress.hpp"
#include "halo/models/qwen4exp/kernels/verify_kernels.hpp"
#include "halo/models/qwen4exp/kernels/dec_moe.hpp"
#include "halo/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace halo::core {
using namespace verify_detail;

bool Verifier::dm_init(std::string& err) {
    using namespace halo::kernels;
    const ModelGeometry& g = *g_;
    const auto& lay = halo::kernels::cpu::expert_layout();
    const char* why = nullptr;
    if (!lay.native) why = "the experts are not a native pack";
    else if (all_resident_) why = "every expert is resident";
    else if (hits_.h_res == nullptr) why = "no host residency table";
    else if ((int64_t) max_t_ * ss_->k > kDmCap) why = "the window is too large";
    for (int64_t l = 0; why == nullptr && l < g.n_layers; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff, f.up_type)) why = "an expert format the kernels do not cover";
    }
    if (why != nullptr) {
        halo::log::warn("halo verify", "the device-planned MoE tail is off: %s", why);
        if (route_tail_skip_ > 0) halo::log::info("halo verify", "route-tail-skip is off: it needs the device-planned MoE tail");
        if (route_prior_ > 0.0f) halo::log::info("halo verify", "route-prior is off: it needs the device-planned MoE tail");
        route_tail_skip_ = 0;
        route_prior_ = 0.0f;
        return true;
    }
    DecMoe* d = new DecMoe;
    dm_ = d;
    const int64_t LE = g.n_layers * g.n_expert, N = g.n_embd, T = max_t_, K = ss_->k;
    d->rings = le_ - lb_;
    d->x_stride = (size_t) (T * N);
    d->xq_stride = (size_t) (T * (N / 32) * 36);
    bool ok = cudaMalloc((void**) &d->res0, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->plan, sizeof(DmPlan)) == cudaSuccess &&
              cudaMalloc((void**) &d->ctr, 64) == cudaSuccess && cudaMemset(d->ctr, 0, 64) == cudaSuccess &&
              mapped((size_t) d->rings * sizeof(DmRing), (void**) &d->h_ring, (void**) &d->m_ring) &&
              mapped((size_t) d->rings * d->x_stride * 4, (void**) &d->h_x, (void**) &d->m_x) &&
              mapped((size_t) d->rings * d->xq_stride, (void**) &d->h_xq, (void**) &d->m_xq) &&
              mapped(64, (void**) &d->h_stop, (void**) &d->m_stop) &&
              cudaMalloc(&d->vscr, dec_moe_vram_scratch_bytes((int) lay.fmt[0].n_ff)) == cudaSuccess &&
              cudaMalloc((void**) &d->vsync, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMemset(d->vsync, 0, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMalloc(&d->xq_pub, d->xq_stride) == cudaSuccess &&
              cudaMalloc((void**) &d->pub, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMemset(d->pub, 0, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMalloc((void**) &d->cseen, 4) == cudaSuccess && cudaMemset(d->cseen, 0, 4) == cudaSuccess &&
              cudaMalloc((void**) &d->landed, (size_t) (d->rings + 1) * 4) == cudaSuccess &&
              cudaMemset(d->landed, 0, (size_t) (d->rings + 1) * 4) == cudaSuccess;
    if (ok && cudaHostAlloc((void**) &d->landed_pin, DecMoe::kPinSlots * 4, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->landed_pin = nullptr;
    }
    d->vram = -1;
    if (ok && hits_.slot_off != nullptr && hits_.n_slots > 0)
        ok = cudaMalloc((void**) &d->slot_off, (size_t) hits_.n_slots * 8) == cudaSuccess &&
             cudaMemcpy(d->slot_off, hits_.slot_off, (size_t) hits_.n_slots * 8, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok)
        ok = cudaMalloc((void**) &d->ptab0, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab0, 0, (size_t) LE * 8) == cudaSuccess;
    if (ok && diagnostics().wait_stats)
        ok = cudaMalloc((void**) &d->wstat, kDmStatWords * 8) == cudaSuccess &&
             cudaMemset(d->wstat, 0, kDmStatWords * 8) == cudaSuccess;
    if (ok && (route_tail_skip_ > 0 || route_prior_ > 0.0f))
        ok = cudaMalloc((void**) &d->route_counts, 8 * 8) == cudaSuccess && cudaMemset(d->route_counts, 0, 8 * 8) == cudaSuccess;
    if (!ok) { err = "verify: device-planned MoE buffers"; return false; }
    d->tags = d->wstat == nullptr;
    d->shadow0.assign((size_t) LE, INT32_MIN);
    d->pin_stride = (size_t) LE * 12;   // [res0 | ptab0] per slot
    if (cudaHostAlloc((void**) &d->pin, d->pin_stride * DecMoe::kPinSlots, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->pin = nullptr;
    }
    // the CPU pool's warm-up predictor, quantized once: every layer's router in 4-bit
    const bool cpu_warm = warm_pool_ != nullptr && warm_src_ != nullptr && K <= 16 &&
                          std::all_of(lay.fmt.begin(), lay.fmt.end(), [&](const halo::kernels::cpu::NativeFmt& f) {
                              return f.bytes == lay.fmt[0].bytes && f.down_off == lay.fmt[0].down_off;
                          });
    std::vector<uint8_t> rq4_h;
    std::vector<float> rs4_h;
    const bool codes_ok = cpu_warm && router_codes(*wt_, g, &rq4_h, &rs4_h);
    if (cpu_warm && codes_ok) {
        CpuWarm* w = new CpuWarm;
        w->rq = std::move(rq4_h);
        w->rs = std::move(rs4_h);
        w->n_expert = (int) g.n_expert; w->n_embd = (int) N; w->k = (int) K; w->src = warm_src_;
        w->bytes = lay.fmt[0].bytes;
        w->down_off = lay.fmt[0].down_off;
        w->logits.assign((size_t) kVerifyMaxT * g.n_expert, 0.0f);
        d->warm = w;
    }
    // The staged share: on one card, a batch window's layer with enough CPU misses sends some of them to CUDA0, copied
    // over PCIe from their RAM copies into staging while the pool reads the rest.  An expert without a mapped RAM copy
    // is never staged; the addresses are read again before a batch window when the source's places moved (a swap).
    // The count staged for m misses balances the pool's time for the rest against the copy's, both reading host
    // memory at once.
    bool stage = warm_src_ != nullptr && staging_ != nullptr && warm_src_->pcie_layer(0) && lay.fmt[0].bytes % 16 == 0;
    for (int64_t l = 1; stage && l < g.n_layers; ++l) stage = lay.fmt[(size_t) l].bytes == lay.fmt[0].bytes;
    const int64_t blob = (int64_t) lay.fmt[0].bytes, stride = dec_moe_stage_stride(blob);
    const int stage_slots =
        (int) std::min<int64_t>(kDmStageMax, (int64_t) (kStagingBlobs * (int64_t) lay.max_blob) / stride);
    if (stage && stage_slots >= 1) {
        // per layer: the pool's experts after its doorbell, the staged ones after the copy, their own launches and
        // the combine's wait for them
        constexpr double kPoolBytesPerUs = 45e3, kLinkBytesPerUs = 18e3, kPoolFixUs = 15.0, kStageFixUs = 35.0;
        const double cpu_us = (double) blob / kPoolBytesPerUs, link_us = (double) blob / kLinkBytesPerUs;
        std::vector<int32_t> kt((size_t) kDmCap + 1, 0);
        for (int m = 1; m <= kDmCap; ++m) {
            double best = kPoolFixUs + m * cpu_us;
            for (int k = 1; k <= std::min(m, stage_slots); ++k) {
                const double t = std::max(kPoolFixUs + (m - k) * cpu_us, kStageFixUs + k * link_us);
                if (t < best) { best = t; kt[(size_t) m] = k; }
            }
        }
        int st_least = 0, st_greatest = 0;   // the copy's blocks go first as the FFN read's leave the SMs
        bool sok = cudaMalloc((void**) &d->stage_src, (size_t) LE * 8) == cudaSuccess &&
                   cudaMemset(d->stage_src, 0, (size_t) LE * 8) == cudaSuccess &&
                   cudaMallocHost((void**) &d->stage_host, (size_t) LE * 8) == cudaSuccess &&
                   cudaMalloc((void**) &d->stage_k, kt.size() * 4) == cudaSuccess &&
                   cudaMemcpy(d->stage_k, kt.data(), kt.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                   cudaDeviceGetStreamPriorityRange(&st_least, &st_greatest) == cudaSuccess &&
                   cudaStreamCreateWithPriority(&d->st_s, cudaStreamNonBlocking, st_greatest) == cudaSuccess &&
                   cudaEventCreateWithFlags(&d->st_fork, cudaEventDisableTiming) == cudaSuccess &&
                   cudaEventCreateWithFlags(&d->st_join, cudaEventDisableTiming) == cudaSuccess;
        if (!sok) { err = "verify: the staged share's buffers"; return false; }
        std::memset(d->stage_host, 0, (size_t) LE * 8);
        d->stage_gen = warm_src_->places_generation() + 1;   // read at the first batch window
        d->stage_stride = stride;
        d->stage_blocks = 8;
        d->stage_min = 1;
        while (d->stage_min < kDmCap && kt[(size_t) d->stage_min] == 0) ++d->stage_min;
    }
    halo::log::info("halo verify", "the MoE tail planned on the device");
    if (d->warm != nullptr)
        halo::log::info("halo verify", "the CPU pool warms its likeliest next-layer expert between doorbells");
    return true;
}

void Verifier::dm_free() {
    DecMoe* d = dm_;
    if (d == nullptr) return;
    if (d->warm != nullptr) {
        warm_pool_->idle_stop();
        delete d->warm;
    }
    if (d->st_s) { cudaStreamSynchronize(d->st_s); cudaStreamDestroy(d->st_s); }
    for (cudaEvent_t e : {d->st_fork, d->st_join})
        if (e) cudaEventDestroy(e);
    for (void* p : {(void*) d->res0, (void*) d->plan, (void*) d->ctr, (void*) d->slot_off, d->vscr,
                    (void*) d->vsync, d->xq_pub, (void*) d->wstat,
                    (void*) d->ptab0, (void*) d->pub, (void*) d->cseen, (void*) d->landed,
                    (void*) d->stage_src, (void*) d->stage_k, (void*) d->route_counts})
        if (p) cudaFree(p);
    if (d->stage_host) cudaFreeHost(d->stage_host);
    if (d->pin) cudaFreeHost(d->pin);
    if (d->landed_pin) cudaFreeHost(d->landed_pin);
    for (void* h : {(void*) d->h_ring, (void*) d->h_x, (void*) d->h_xq, (void*) d->h_stop})
        if (h) cudaFreeHost(h);
    delete d;
    dm_ = nullptr;
}

std::vector<std::string> Verifier::route_report() const {
    if (dm_ == nullptr || dm_->route_counts == nullptr) return {};
    const OnDevice on(device_);
    unsigned long long c[8] = {};
    if (cudaMemcpy(c, dm_->route_counts, sizeof c, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    std::vector<std::string> out;
    char b[240];
    if (route_tail_skip_ > 0 && c[0] > 0) {
        std::snprintf(b, sizeof b, "route-tail-skip %d: %llu of %llu routed entries no GPU held dropped (%.1f%%); %.1f%% of all %llu entries",
                      route_tail_skip_, c[2], c[1], c[1] > 0 ? 100.0 * (double) c[2] / (double) c[1] : 0.0,
                      100.0 * (double) c[2] / (double) c[0], c[0]);
        out.push_back(b);
    }
    if (route_prior_ > 0.0f && c[3] > 0) {
        std::snprintf(b, sizeof b, "route-prior %g: %.1f%% of %llu routed entries picked because of the bias; %.1f%% of entries held by no GPU",
                      (double) route_prior_, 100.0 * (double) c[4] / (double) c[3], c[3],
                      100.0 * (double) c[5] / (double) c[3]);
        out.push_back(b);
    }
    return out;
}

}  // namespace halo::core
