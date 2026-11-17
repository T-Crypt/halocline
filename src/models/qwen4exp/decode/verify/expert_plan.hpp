#pragma once

#include "common.hpp"
#include "cpu_warm.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include <vector>

namespace basalt::core {

// the MoE tail (dec_moe.hpp): planned on CUDA0 from device residency tables, the CPU pool served by the host loop
// (only its own entries).
struct DecMoe {
    int64_t rings = 0;                       ///< rings per window (one per layer)
    int32_t* res0 = nullptr;                 ///< CUDA0: [n_layers * n_expert] CUDA0 slot or -1 (host_res mirrored)
    std::vector<int32_t> shadow0;            ///< what res0 holds
    CpuWarm* warm = nullptr;                 ///< the CPU tier's warm-up (null: off)
    basalt::kernels::DmPlan* plan = nullptr;
    unsigned int* ctr = nullptr;
    unsigned long long* slot_off = nullptr;  ///< CUDA0's per-slot offsets (null: slot * blob)
    int vram = -1;   // the VRAM share's kernels (dec_moe_vram's kind; -1 grouped)
    void* vscr = nullptr;                    ///< its gate/up rows
    void* xq_pub = nullptr;   // the side stream's q8_1 rows for the ring
    unsigned int* vsync = nullptr;           ///< its counters
    unsigned long long* wstat = nullptr;   // the per-layer wait counters (device), with --wait-stats
    unsigned long long* ptab0 = nullptr;   // [n_layers * n_expert] CUDA0 blob address or 0
    unsigned int* pub = nullptr;           // [rings][4] the FFN read's counts for the ring's block (HcFusedArgs::pub_ctr)
    unsigned int* cseen = nullptr;         // the combine's last ring whose CPU word it saw (dec_moe_combine's cpu_seen)
    unsigned long long* route_counts = nullptr;   // [8] what the opt-in routing changes did (route_report); null: none on
    // [rings + 1]: per layer the last adaptive round whose copies into its slots are in, then the round the window's
    // gate/up launches wait for (DmGlueArgs::landed)
    unsigned int* landed = nullptr;
    uint32_t* landed_pin = nullptr;        // pinned: the need copies' sources, a ring of kPinSlots
    int landed_i = 0;
    int64_t wstat_windows = 0, wstat_t = 0;
    // mapped: per ring header, FFN input and q8_1 rows; [0] the peer's ring, [16] its done ring, [32] its abort
    // record (kDmDiagWords); the peer's rows
    basalt::kernels::DmRing* h_ring = nullptr; basalt::kernels::DmRing* m_ring = nullptr;
    float* h_x = nullptr; float* m_x = nullptr;
    uint8_t* h_xq = nullptr; uint8_t* m_xq = nullptr;
    uint32_t* h_w = nullptr; uint32_t* m_w = nullptr;
    float* h_rows = nullptr; float* m_rows = nullptr;
    bool tag = false;       ///< the combine polls the peer's rows (each dword not yet written holds the tag word)
    bool tags = false;      ///< the combine waits for the CPU pool's word itself: no wait launch (without --wait-stats)
    size_t rows_words = 0;
    bool tag_dirty = false; ///< a released window may have left written rows behind
    size_t x_stride = 0, xq_stride = 0;      ///< per ring, floats / bytes
    // the staged share: every expert's host-RAM address (device alias) or 0, its host copy and
    // the source's places it was read at, the staged count per count of distinct CPU misses, the copy's own stream
    unsigned long long* stage_src = nullptr;
    unsigned long long* stage_host = nullptr;
    uint64_t stage_gen = 0;
    int32_t* stage_k = nullptr;
    cudaStream_t st_s = nullptr;
    cudaEvent_t st_fork = nullptr, st_join = nullptr;
    int64_t stage_stride = 0;
    int stage_blocks = 0;
    int stage_min = 0;   // the fewest distinct CPU misses a layer stages a share of
    // pinned staging of the residency uploads: the copies take their source from here, so the call does
    // not wait for the stream; a ring of slots, each reused only after the window's stream sync
    uint8_t* pin = nullptr;
    size_t pin_stride = 0;
    int pin_i = 0;
    static constexpr int kPinSlots = 8;
};

}  // namespace basalt::core
