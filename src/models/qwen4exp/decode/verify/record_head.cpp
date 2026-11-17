#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "halo/models/qwen4exp/decode/verify.hpp"
#include "halo/models/qwen4exp/decode/native_head.hpp"
#include "halo/core/on_device.hpp"
#include "halo/models/qwen4exp/kernels/fused_gr.hpp"
#include "halo/models/qwen4exp/kernels/hc_fused.hpp"
#include "halo/models/qwen4exp/kernels/hc_read_ada.hpp"
#include "halo/models/qwen4exp/kernels/cvec.hpp"
#include "halo/kernels/native_mmvq.hpp"
#include "halo/models/qwen4exp/kernels/q5k_head.hpp"
#include "halo/models/qwen4exp/kernels/q6k_head.hpp"
#include "halo/models/qwen4exp/kernels/q80_vhead.hpp"
#include "halo/kernels/sampler.hpp"
#include <algorithm>
#include <exception>
#include <string>

namespace halo::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::head() {
    using namespace halo::kernels;
    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = v_.bo_ + t * N; a.inj_prev = v_.inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = v_.lo_ + t * g.hc_lr;
                a.rs = v_.rs_ + t * HC;
                a.inject_out = v_.head_inj_;
                a.mixed = v_.head_mixed_ + t * N;
            }
            if (dec_hc) {
                HcFusedArgs h;
                h.T = T; h.R = Rt(0); h.R_out = Rt(0); h.apply = true; h.bo_prev = v_.bo_; h.inj_prev = v_.inj2_;
                h.w_norm = (const float*) hn->data; h.w_down = (const uint16_t*) hd->data; h.w_up = (const uint16_t*) hu->data;
                h.eps = EPS; h.lo = v_.lo_; h.rs = v_.rs_; h.mixed = v_.head_mixed_; h.xn = v_.xn_;
                hc_fused_read(h, cs);
            } else if (!v_.prof_on_ && !v_.batch_rec_ && !ple_open && !lz_open && !l2a_open && hc_read_ada_supported(fa, T)) {
                hc_read_ada(fa, T, cs);
            } else {
                fused_gr_read_multi(fa, T, v_.xn_, cs);
            }
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = v_.head_mixed_ + t * N;
                if (v_.head_ != nullptr && v_.head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, v_.head_logits_ + (size_t) t * v_.n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (v_.head_ != nullptr && v_.head_->loaded()) {
            try {
                native_quantize_q8_1(v_.head_mixed_, v_.xq_, (int) N, T, cs);
                if (v_.head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits, one wave
                    q5k_head_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                else if (v_.head_->type() == 14 && v_.vhead_scratch_ != nullptr && q6k_vhead_supported((int) N, (int) v_.n_vocab_, T))
                    q6k_vhead(v_.head_->vproxy(), v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T,
                              v_.vhead_scratch_, cs, v_.batch_rec_ ? nullptr : v_.head_from_d_);
                else if (v_.head_->type() == 14 && q6k_head_supported((int) N, T))
                    q6k_head_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                // the exact list's rows, the rest bounded (a batch window's list length is 0: every row is scored)
                else if (v_.head_->planes() && v_.head_keep_d_ != nullptr)
                    q80p_vhead(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T,
                               v_.head_keep_d_, v_.vhead_scratch_, cs);
                else if (v_.head_->planes())
                    q80p_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                else
                    native_mmvq(v_.head_->type(), v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(v_.head_logits_, T, (int) v_.n_vocab_, nullptr, 0, sp, v_.m_out_, cs);
    }
    stamp(g.n_layers, 1);
    return true;
}

}  // namespace halo::core
