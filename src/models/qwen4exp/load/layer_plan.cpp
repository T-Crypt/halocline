#include "halo/models/qwen4exp/shared_names.hpp"
#include "context.hpp"

namespace halo::models::qwen4exp::generate {

int layer_plan(Context& context) {
    auto& o = *context.o;
    auto& pack_files = *context.pack_files;
    auto& err = *context.err;
    auto& skip = *context.skip;

    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache is sized
    // first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`), so it withholds
    // no VRAM of its own.  `plan_lend` uses the exact `Prefill::bytes_needed` as soon as it can.
    const bool pf_borrow = !o.expert_profile.empty();
    const int64_t kWindowMib = 96;       // a server's verify windows, allocated after the cache is sized, with margin
    // the dense weights of layers outside [lb, le) (the PLE tensors stay), for a part of the model on another card
    const std::set<std::string> skip_base = skip;
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) return;
            const int64_t l = std::atoll(n.c_str() + 4);
            if (l < lb || l >= le) out.insert(n);
        });
    };
    // The hyper-connection down/up matrices are held only as 8-bit copies - the arena leaves their bf16 form
    // out and they are staged, copied and dropped right after the load
    std::set<std::string> hc8_names;
    if (halo::kernels::hc8_only()) {
        for_each_index_name(pack_files.index, [&](const std::string& n) {
            const bool blk = n.rfind("blk.", 0) == 0 &&
                             (n.ends_with(".hc_attn_down.weight") || n.ends_with(".hc_attn_up.weight") ||
                              n.ends_with(".hc_ffn_down.weight") || n.ends_with(".hc_ffn_up.weight"));
            if (blk || n == "output_hc_down.weight" || n == "output_hc_up.weight") hc8_names.insert(n);
        });
        for (const std::string& n : hc8_names) skip.insert(n);
    }
    uint64_t pool_bytes = 0;
    if (!halo::core::WeightTable::pool_bytes(pack_files, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        return fatal_start_error(err);
    }

    context.pf_borrow = &pf_borrow;
    context.kWindowMib = &kWindowMib;
    context.skip_base = &skip_base;
    context.add_foreign = add_foreign;
    context.hc8_names = &hc8_names;
    context.pool_bytes = &pool_bytes;
    return weights(context);
}

}
