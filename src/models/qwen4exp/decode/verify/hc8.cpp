#include "common.hpp"
#include "halo/core/load_progress.hpp"
#include "halo/models/qwen4exp/decode/verify.hpp"
#include "halo/models/qwen4exp/decode/native_head.hpp"
#include "halo/core/on_device.hpp"
#include "halo/models/qwen4exp/kernels/hc_fused.hpp"
#include "halo/kernels/pdl.hpp"
#include "halo/core/progress.hpp"
#include "halo/log.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace halo::core {
using namespace verify_detail;

bool hc8_table_has(const std::string& name, int rows, int cols) { return halo::core::hc8_find(name, rows, cols) != nullptr; }

const void* hc8_table_adopt(const std::string& name, const uint16_t* w, int rows, int cols) {
    const Hc8Copy* c = hc8_find(name, rows, cols);
    if (c == nullptr) return nullptr;
    return halo::kernels::hc8_upload(w, c->q(), c->s(), rows, cols);
}

namespace {
/// The hyper-connection matrices, by name: (name, rows, cols).
struct Hc8Tensor { std::string name; int rows, cols; };
std::vector<Hc8Tensor> hc8_tensors(const ModelGeometry& g) {
    const int LR = (int) g.hc_lr, D = (int) (g.hc * g.n_embd), HCn = (int) g.hc;
    std::vector<Hc8Tensor> v;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string blk = "blk." + std::to_string(l) + ".";
        for (const char* p : {"hc_attn_", "hc_ffn_"}) {
            const std::string s(p);
            v.push_back({blk + s + "down.weight", LR, D});
            v.push_back({blk + s + "up.weight", D, LR});
            v.push_back({blk + s + "inject.weight", HCn, D});
        }
    }
    v.push_back({"output_hc_down.weight", LR, D});
    v.push_back({"output_hc_up.weight", D, LR});
    return v;
}
// the copies' size: the int8 codes and one fp16 scale per 32 values
uint64_t hc8_copy_bytes(const Hc8Tensor& t) { return (uint64_t) t.rows * (uint64_t) t.cols * 9 / 8; }
}  // namespace

uint64_t Verifier::hc8_plan_bytes(const std::function<bool(const std::string&)>& has, const ModelGeometry& g) {
    if (!halo::kernels::hc8_enabled()) return 0;
    uint64_t total = 0;
    for (const Hc8Tensor& t : hc8_tensors(g))
        if (has(t.name)) total += hc8_copy_bytes(t);
    return total;
}

bool Verifier::prepare_hc8(const WeightTable& wt, const ModelGeometry& g, std::string& err) {
    if (!halo::kernels::hc8_enabled()) return true;
    const std::vector<Hc8Tensor> tensors = hc8_tensors(g);
    const uint64_t total = hc8_plan_bytes([&](const std::string& n) { return wt.find(n) != nullptr; }, g);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    LoadProgress progress(LoadProgress::cuda(dev), total);
    for (const Hc8Tensor& t : tensors) {
        const WeightRef* r = wt.find(t.name);
        if (r == nullptr || r->data == nullptr) continue;   // not held on this device
        if (hc8_table_adopt(t.name, (const uint16_t*) r->data, t.rows, t.cols) == nullptr) {
            err = "hc8: no 8-bit copy of " + t.name;
            return false;
        }
        progress.add(hc8_copy_bytes(t));
    }
    progress.finish();
    halo::log::info("halo hc8", "8-bit hyper-connection copies, %.0f MiB of VRAM",
                 (double) halo::kernels::hc8_bytes() / 1048576.0);
    return true;
}

}  // namespace halo::core
