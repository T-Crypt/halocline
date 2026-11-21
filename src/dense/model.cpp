// src/dense/model.cpp - the dense stack's shared scratch and geometry checks.
#include "internal.hpp"

#include "halo/kernels/gdn_prompt.hpp"
#include "halo/kernels/gqa_attn.hpp"
#include "halo/kernels/head_rows.hpp"
#include "halo/kernels/iq_kernels.hpp"
#include "halo/kernels/kv_q4_layout.hpp"
#include "halo/kernels/logprobs.hpp"
#include "halo/kernels/nvfp4_planes.hpp"
#include "halo/prefill/fp4_dense.hpp"
#include "halo/prefill/fp8_gemm.hpp"
#include "halo/kernels/native_mmvq.hpp"
#include "halo/log.hpp"
#include "halo/vision/encoder_cpu.hpp"
#include "halo/vision/encoder_cuda.hpp"
#include "halo/vision/mmproj.hpp"

#include <algorithm>
#include <thread>
#include <cstdlib>
#include <fstream>

namespace halo::dense {

Arena::~Arena() {
    if (base_) cudaFree(base_);
}

bool Arena::reserve(size_t bytes) {
    if (base_) cudaFree(base_);
    base_ = nullptr;
    size_ = used_ = 0;
    if (bytes == 0) return true;
    if (cudaMalloc(&base_, bytes) != cudaSuccess) {
        cudaGetLastError();
        base_ = nullptr;
        return false;
    }
    size_ = bytes;
    return true;
}

void* Arena::take_bytes(size_t n) {
    const size_t at = (used_ + 255) & ~(size_t) 255;
    if (at + n > size_) return nullptr;
    used_ = at + n;
    return base_ + at;
}

int Geometry::gdn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Gdn); }
int Geometry::attn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Attention); }

std::string Geometry::check() const {
    if ((int) kinds.size() != n_layers || n_layers <= 0) return "the layer kinds do not cover the layers";
    if (n_embd % 256 || n_ff % 256) return "widths must be whole 256-value blocks";
    if (attn_layers() > 0) {
        if (head_dim != 256) return "attention heads must be 256 wide";
        if (n_kv <= 0 || n_head % n_kv || n_head / n_kv > 8) return "attention head grouping not supported";
        if (rope.rot <= 0 || rope.rot > head_dim || rope.rot % 2 || rope.rot / 2 > 64) return "rotary width not supported";
    }
    if (gdn_layers() > 0) {
        if (!kernels::gdn_prompt_supported(gdn_state, gdn_k_heads, gdn_v_heads, gdn_conv)) return "GDN geometry not supported";
        if (!kernels::dec_gdn_rec_supported(n_embd)) return "GDN window kernel not built for this width";
    }
    return {};
}

namespace {

// bytes of the decode window's scratch
// the block drafter's rows in the window scratch
size_t block_scratch_floats(const Geometry& g, const BlockDrafter& bd) {
    if (!bd.present) return 0;
    const size_t W = kBatch, AQ = (size_t) bd.n_head * bd.head_dim, KV = (size_t) bd.n_kv * bd.head_dim;
    return W * (3 * AQ + 2 * KV + bd.layers[0].conv_a.rows + g.n_embd + kernels::kLogprobStride + bd.rank + 3) +
           kernels::gqa_decode_scratch_floats(bd.window + 256, kWindow, bd.n_head, bd.head_dim) + 64 * 64;
}

int64_t widest_input(const Geometry& g) {
    const size_t AQ = (size_t) g.n_head * g.head_dim;
    return (int64_t) std::max<size_t>({(size_t) g.n_embd, (size_t) g.n_ff, (size_t) g.value_width(), AQ, 2 * (size_t) g.n_embd});
}
// a window's activation buffers for the widest input: the window gemm's bf16 rows, MMQ's q8_1 rows
size_t act_slot(const Geometry& g) { return (kernels::wgemm::act_bytes(widest_input(g)) + 255) / 256 * 256; }
size_t q8_slot(const Geometry& g) { return (prefill::mmq::q8_bytes(kBatch, widest_input(g)) + 255) / 256 * 256; }

// The window gemm's split-K workspace for every call a window makes (forward.cpp): the groups and each matrix alone, in
// either format (a superset is harmless).
void wgemm_need(const Geometry& g, size_t& partial, size_t& tickets) {
    namespace wg = kernels::wgemm;
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const int64_t C = g.conv_channels(), VD = g.value_width(), F = g.n_ff;
    struct Shape {
        int64_t rows[3];
        int n;
        int64_t K;
        wg::Epilogue epi;
    };
    std::vector<Shape> shapes = {{{2 * AQ, KV, KV}, 3, H, wg::Epilogue::Store}, {{2 * AQ}, 1, H, wg::Epilogue::Store},
                                 {{KV}, 1, H, wg::Epilogue::Store},          {{H}, 1, AQ, wg::Epilogue::Add},
                                 {{F, F}, 2, H, wg::Epilogue::SwiGLU},       {{F}, 1, H, wg::Epilogue::Store},
                                 {{H}, 1, F, wg::Epilogue::Add},             {{g.n_vocab}, 1, H, wg::Epilogue::Store},
                                 {{256}, 1, H, wg::Epilogue::Store},   // the block drafter's selector (its rank)
                                 {{kDraftVocab}, 1, H, wg::Epilogue::Store}};   // and its head's rows
    if (g.gdn_layers() > 0) {
        shapes.push_back({{C, VD}, 2, H, wg::Epilogue::Store});
        shapes.push_back({{C}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{VD}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{H}, 1, VD, wg::Epilogue::Add});
    }
    partial = tickets = 0;
    for (const Shape& sh : shapes)
        for (const wg::Format f : {wg::Format::F8, wg::Format::Nvfp4, wg::Format::Bf16}) {
            if (sh.K <= 0 || sh.K % 256 != 0) continue;
            wg::Call c;
            c.fmt = f;
            c.K = sh.K;
            c.n = sh.n;
            c.epi = sh.epi;
            bool ok = true;
            for (int i = 0; i < sh.n; ++i) {
                c.m[i].rows = sh.rows[i];
                ok = ok && sh.rows[i] > 0 && (f == wg::Format::F8 || sh.rows[i] % 128 == 0);
            }
            if (!ok || (f != wg::Format::Nvfp4 && sh.epi == wg::Epilogue::SwiGLU)) continue;
            size_t p, t;
            wg::workspace_need(c, p, t);
            partial = std::max(partial, p);
            tickets = std::max(tickets, t);
        }
}

size_t window_bytes(const Geometry& g, const BlockDrafter& bd, int64_t max_context, bool has_wg) {
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim;
    size_t f = 4 * W * H + W * g.value_width() + W * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) + 3 * W * g.n_ff;
    if (g.attn_layers() > 0) f += kernels::gqa_decode_scratch_floats(max_context, kWindow, g.n_head);
    f += W * (2 * (size_t) g.conv_channels() + 2 * (size_t) g.gdn_v_heads) + (sizeof(BatchTable) + 3) / 4;
    f += W * 2 * H + kernels::row_logprobs_scratch_floats(kWindow, g.n_vocab) + W * g.value_width() + W * (size_t) g.n_vocab;
    f += block_scratch_floats(g, bd);
    // the window gemm's activation slots and split-K sums, taken when a product runs on it (F8 or NVFP4 planes, or the
    // block drafter, whose products load as planes); a pack whose matrices are all GGUF quants runs every product on MMQ
    size_t wg = 0;
    if (has_wg || bd.present) {
        size_t wp, wt;
        wgemm_need(g, wp, wt);
        wg = 2 * act_slot(g) + (wp + wt + 1) * 4;
    }
    return f * 4 + wg + q8_slot(g) + W * 4 * 6 + 76 * 256;
}

size_t chunk_bytes(const Geometry& g, bool has_fp4, Drafter drafter) {
    const size_t T = kChunk, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, C = (size_t) g.conv_channels();
    const size_t VD = (size_t) g.value_width(), HV = (size_t) g.gdn_v_heads;
    const size_t widest = std::max<size_t>({H, (size_t) g.n_ff, VD, AQ});
    // the residual buffers the chunk's drafter takes: x, xn and tmp always; the MTP history adds cat and hn (its T + 1
    // rows, hence H * 4), the block drafter's feature rows dacc
    const size_t res = 3 + (drafter == Drafter::Mtp ? 2 : drafter == Drafter::Block ? 1 : 0);
    const size_t keep = (res * T * H + (drafter == Drafter::Mtp ? H : 0)) * 4 +
                        std::max(prefill::mmq::q8_bytes(T, (int64_t) widest), prefill::fp8::act_bytes(T, (int64_t) widest)) +
                        (has_fp4 ? prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) widest) : 0) + 2 * T * 4 + 5 * 256;
    const size_t gdn = g.gdn_layers() ? (2 * T * C + 3 * T * VD + 4 * T * HV) * 4 + kernels::gdn_work_bytes(T) : 0;
    const size_t attn = g.attn_layers() ? T * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) * 4 +
                                              kernels::gqa_prompt_split_floats((int64_t) T, g.n_head, kAttnSplits) * 4
                                        : 0;
    const size_t ffn = 3 * T * g.n_ff * 4;
    return keep + std::max({gdn, attn, ffn}) + T * 4 * 6 + 32 * 256;
}

}  // namespace

namespace {
// The block drafter as decode windows run it, on the window gemm: q, o, gate, up and down rounded to NVFP4 planes in place
// and the convolutions repacked to Q8 planes in place (prompt chunks never read them); k and v (NVFP4) and the feature
// projection (Q8 planes), which prompt chunks read as blocks, get copies.  Then the window gemm's workspace grows to its
// calls if they need more.
bool decode_drafter(Model& m, std::string& err) {
    m.bd_dec = m.w.dflash;
    BlockDrafter& bd = m.bd_dec;
    if (!bd.present || m.o.drafter != Drafter::Block) return true;
    void* tmp = nullptr;
    size_t tmp_bytes = 0;
    auto q8 = [](const Linear& l) { return l.type == 8 && l.cols % 256 == 0 && l.rows > 0; };
    auto bytes = [](const Linear& l) { return (size_t) l.rows * (size_t) l.cols / 32 * 34; };
    auto in_place = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        const size_t n = bytes(l);
        if (n > tmp_bytes) {
            if (tmp) cudaFree(tmp);
            tmp = nullptr;
            if (cudaMalloc(&tmp, n) != cudaSuccess) return false;
            tmp_bytes = n;
        }
        if (cudaMemcpy(tmp, l.w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
        kernels::wgemm::q8_planes(tmp, l.rows, l.cols, const_cast<void*>(l.w), nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        l.type = kTypeQ8P;
        return true;
    };
    auto copy = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes(l)) != cudaSuccess) return false;
        m.owned.push_back(p);
        kernels::wgemm::q8_planes(l.w, l.rows, l.cols, p, nullptr);
        l.w = p;
        l.type = kTypeQ8P;
        return true;
    };
    // The drafter's big products (gate, up, down, q, o) in NVFP4 planes, rounded from its Q8_0 at load: the drafts' precision
    // only (every draft is verified exactly, so the output does not change); about half the bytes a pass reads.
    auto to_fp4 = [&](Linear& l) -> bool {
        if (!q8(l) || !kernels::nvfp4p::fits(l.rows, l.cols)) return in_place(l);
        const size_t n = bytes(l);
        if (n > tmp_bytes) {
            if (tmp) cudaFree(tmp);
            tmp = nullptr;
            if (cudaMalloc(&tmp, n) != cudaSuccess) return false;
            tmp_bytes = n;
        }
        void* sc = nullptr;
        if (cudaMemcpy(tmp, l.w, n, cudaMemcpyDeviceToDevice) != cudaSuccess || cudaMalloc(&sc, 16) != cudaSuccess) return false;
        m.owned.push_back(sc);
        kernels::nvfp4p::from_q8(tmp, l.rows, l.cols, const_cast<void*>(l.w), (float*) sc, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        l.scale = (const float*) sc;
        l.type = kTypeNvfp4Planes;
        return true;
    };
    bool ok = true;
    for (BlockDrafter::Layer& D : bd.layers) {
        for (Linear* l : {&D.gate, &D.up, &D.down, &D.q, &D.o}) ok = ok && to_fp4(*l);
        for (Linear* l : {&D.conv_a, &D.conv_f}) ok = ok && in_place(*l);
        if (kernels::nvfp4p::fits(D.k.rows, D.k.cols) && kernels::nvfp4p::fits(D.v.rows, D.v.cols)) {
            // k and v as NVFP4 copies (prompt chunks read their blocks), so q, k and v stay one call
            for (Linear* l : {&D.k, &D.v}) {
                void* p = nullptr;
                void* sc = nullptr;
                if (!q8(*l) || cudaMalloc(&p, kernels::nvfp4p::bytes(l->rows, l->cols)) != cudaSuccess ||
                    cudaMalloc(&sc, 16) != cudaSuccess) {
                    ok = false;
                    break;
                }
                m.owned.push_back(p);
                m.owned.push_back(sc);
                kernels::nvfp4p::from_q8(l->w, l->rows, l->cols, p, (float*) sc, nullptr);
                l->w = p;
                l->scale = (const float*) sc;
                l->type = kTypeNvfp4Planes;
            }
        } else {
            ok = ok && copy(D.k) && copy(D.v);
        }
    }
    for (Linear& l : bd.fc) ok = ok && copy(l);
    if (tmp) cudaFree(tmp);
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
        err = m.o.arch + ": laying out the block drafter for decode windows failed";
        return false;
    }
    // the workspace for the drafter's calls (each matrix, q + k + v, k + v)
    namespace wg = kernels::wgemm;
    size_t need_p = m.ws.w8.partial_floats, need_t = m.ws.w8.tickets_n;
    auto need = [&](std::initializer_list<const Linear*> ls) {
        wg::Call c;
        c.fmt = wg::Format::Q8;
        c.n = 0;
        for (const Linear* l : ls) {
            if (l->type != kTypeQ8P && l->type != kTypeNvfp4Planes) return;
            c.fmt = l->type == kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
            c.K = l->cols;
            c.m[c.n++].rows = l->rows;
        }
        if (ls.size() == 2 && c.fmt == wg::Format::Nvfp4) c.epi = wg::Epilogue::SwiGLU;   // the FFN pair
        size_t p, t;
        wg::workspace_need(c, p, t);
        need_p = std::max(need_p, p);
        need_t = std::max(need_t, t);
    };
    for (const BlockDrafter::Layer& D : bd.layers) {
        for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) need({l});
        need({&D.q, &D.k, &D.v});
        need({&D.k, &D.v});
        need({&D.gate, &D.up});
    }
    for (const Linear& l : bd.fc) need({&l});
    if (need_p > m.ws.w8.partial_floats || need_t > m.ws.w8.tickets_n) {
        void *p = nullptr, *t = nullptr;
        if (cudaMalloc(&p, std::max<size_t>(need_p, 1) * 4) != cudaSuccess || cudaMalloc(&t, need_t * 4) != cudaSuccess ||
            cudaMemset(t, 0, need_t * 4) != cudaSuccess) {
            err = m.o.arch + ": out of device memory for the drafter's workspace";
            return false;
        }
        m.owned.push_back(p);
        m.owned.push_back(t);
        m.ws.w8.partial = (float*) p;
        m.ws.w8.tickets = (int32_t*) t;
        m.ws.w8.partial_floats = need_p;
        m.ws.w8.tickets_n = need_t;
    }
    return true;
}

}  // namespace

// a cell of one attention layer, K and V: fp16, int8 codes with their fp16 scales, K that int8 form and V in rotated
// q4_0 blocks, or both in those blocks (kv_q4_layout.hpp: an fp16 scale and 16 code bytes per 32 values)
uint64_t kv_cell_bytes(const Geometry& g, KvFormat fmt) {
    const uint64_t k8 = (uint64_t) g.n_kv * ((uint64_t) g.head_dim + (uint64_t) (g.head_dim / 64) * 2);
    const uint64_t v4 = (uint64_t) g.n_kv * kernels::kv_q4_head_bytes(g.head_dim);
    switch (fmt) {
        case KvFormat::Fp16: return (uint64_t) g.n_kv * g.head_dim * 2 * 2;
        case KvFormat::Int8: return 2 * k8;
        case KvFormat::K8V4: return k8 + v4;
        case KvFormat::Q4: return 2 * v4;
    }
    return 0;
}

int64_t resident_cells(int64_t max_context, int64_t kv_resident) {
    return kv_resident > 0 && kv_resident < max_context ? kv_resident : max_context;
}

ScratchPaths scratch_paths(const Weights& w) {
    ScratchPaths p{false, false};
    const auto see = [&](const Linear& l) {
        if (!l.w) return;
        if (l.type == kTypeF8 || l.type == kTypeNvfp4Planes) p.has_wg = true;
        if (l.type == kTypeNvfp4Planes) p.has_fp4 = true;
    };
    see(w.head);
    for (const Layer& L : w.layers)
        for (const Linear* l : {&L.qkv, &L.z, &L.out, &L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) see(*l);
    if (w.mtp.present) {
        see(w.mtp.fc);
        for (const Linear* l : {&w.mtp.layer.q, &w.mtp.layer.k, &w.mtp.layer.v, &w.mtp.layer.o, &w.mtp.layer.gate,
                                &w.mtp.layer.up, &w.mtp.layer.down})
            see(*l);
    }
    if (w.dflash.present) p.has_wg = p.has_fp4 = true;   // its Q8 matrices load as planes; its k/v chunks read as planes
    return p;
}

uint64_t shared_bytes(const Geometry& g, int64_t max_context, KvFormat kv, int64_t kv_resident, ScratchPaths paths,
                      Drafter drafter) {
    const uint64_t stage = g.attn_layers() > 0 ? (uint64_t) (max_context - resident_cells(max_context, kv_resident)) * kv_cell_bytes(g, kv) : 0;
    return window_bytes(g, BlockDrafter{}, max_context, paths.has_wg) + chunk_bytes(g, paths.has_fp4, drafter) + stage +
           (512ull << 20);
}

uint64_t session_host_bytes(const Geometry& g, int64_t max_context, KvFormat kv, int64_t kv_resident) {
    return (uint64_t) (max_context - resident_cells(max_context, kv_resident)) * g.attn_layers() * kv_cell_bytes(g, kv);
}

uint64_t session_layer_bytes(const Geometry& g, int il, int64_t max_context, KvFormat kv, int64_t kv_resident) {
    if (g.kinds[(size_t) il] == Mixer::Gdn) {
        const uint64_t C = (uint64_t) g.conv_channels();
        return (2 * ((uint64_t) g.gdn_state * g.gdn_v_heads * g.gdn_state + C * 3) + kWindow * (2 * C + 2 * (uint64_t) g.gdn_v_heads)) * 4;
    }
    return (uint64_t) resident_cells(max_context, kv_resident) * kv_cell_bytes(g, kv) + 4 * 256;
}

uint64_t session_bytes(const Geometry& g, int64_t max_context, bool mtp, KvFormat kv, int64_t kv_resident,
                       KvFormat mtp_format, int64_t mtp_window, int64_t draft_rows) {
    uint64_t b = 64 * 256;
    for (int il = 0; il < g.n_layers; ++il) b += session_layer_bytes(g, il, max_context, kv, kv_resident);
    // the drafter's cells: its own layer's format, and with a window only the ring's
    const int64_t cap = mtp ? mtp_ring_cap(max_context, mtp_window) : 0;
    // the drafter's head row (a --draft-vocab subset is that many rows wide), its picks' rows and the zero pick
    const uint64_t head_rows = draft_rows > 0 ? (uint64_t) draft_rows : (uint64_t) g.n_vocab;
    const uint64_t drafter =
        mtp ? (uint64_t) (cap > 0 ? cap : max_context) * kv_cell_bytes(g, mtp_format) +
                  ((uint64_t) (kWindow + 3) * g.n_embd + head_rows + (uint64_t) kWindow * 64 + 1) * 4
            : 0;
    return b + drafter + (uint64_t) kWindow * g.n_vocab * 4;
}

uint64_t draft_head_bytes(const Linear& head, int64_t n) {
    if (n <= 0) return 0;
    const size_t row = kernels::iq_row_bytes(head.type, head.cols);
    return row == 0 ? 0 : (uint64_t) n * row;
}

bool read_draft_vocab(const std::string& path, int64_t n_vocab, std::vector<int32_t>& ids, std::string& err) {
    ids.clear();
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size <= 0 || size % 4 != 0) {
        err = path + ": " + std::to_string((long long) size) + " bytes is not a list of int32 token ids";
        return false;
    }
    f.seekg(0);
    std::vector<int32_t> got((size_t) size / 4);
    if (!f.read(reinterpret_cast<char*>(got.data()), size)) {
        err = path + ": short read";
        return false;
    }
    if (got.size() > (size_t) n_vocab) {
        err = path + ": " + std::to_string((long long) got.size()) + " ids for a vocabulary of " + std::to_string((long long) n_vocab);
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        const int32_t id = got[i];
        if (id < 0 || id >= n_vocab) {
            err = path + ": id " + std::to_string((long long) id) + " at row " + std::to_string((long long) i) +
                  " is outside [0, " + std::to_string((long long) n_vocab) + ")";
            return false;
        }
        if (i > 0 && id <= got[i - 1]) {
            err = path + ": row " + std::to_string((long long) i) + " (id " + std::to_string((long long) id) +
                  ") does not follow id " + std::to_string((long long) got[i - 1]) + ": the list must be strictly ascending";
            return false;
        }
    }
    ids = std::move(got);
    return true;
}

bool gather_draft_head(const Linear& head, const std::vector<int32_t>& ids, void* stream, Linear& out, const int32_t*& map,
                       std::string& err) {
    out = Linear{};
    map = nullptr;
    if (ids.empty()) {
        err = "the draft vocabulary names no tokens";
        return false;
    }
    if (head.w == nullptr || head.rows <= 0 || head.cols <= 0) {
        err = "the pack has no head to gather the draft vocabulary from";
        return false;
    }
    if (head.scale != nullptr || head.row_scales != nullptr) {
        err = "the head carries a tensor scale, so its rows are not whole blocks a gather can copy";
        return false;
    }
    if (!kernels::native_mmvq_supported(head.type)) {
        err = "the head is GGUF type " + std::to_string(head.type) +
              ", which the drafter's one-row product does not read natively";
        return false;
    }
    const size_t row = kernels::iq_row_bytes(head.type, head.cols);
    if (row == 0) {
        err = "the head's rows are not whole blocks of its type";
        return false;
    }
    const uint64_t bytes = (uint64_t) row * ids.size();
    uint8_t* rows = nullptr;
    int32_t* m = nullptr;
    if (cudaMalloc((void**) &rows, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "out of device memory for the draft head (" + std::to_string((unsigned long long) (bytes >> 20)) + " MiB)";
        return false;
    }
    if (cudaMalloc((void**) &m, ids.size() * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(rows);
        err = "out of device memory for the draft vocabulary's id map";
        return false;
    }
    const bool copied = cudaMemcpyAsync(m, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                        (cudaStream_t) stream) == cudaSuccess;
    if (!copied) {
        cudaGetLastError();
        cudaFree(rows);
        cudaFree(m);
        err = "the draft vocabulary's id map upload failed";
        return false;
    }
    kernels::gather_rows((const uint8_t*) head.w, (int64_t) row, m, (int64_t) ids.size(), rows, stream);
    out = head;
    out.w = rows;
    out.rows = (int64_t) ids.size();
    out.scale = nullptr;
    out.row_scales = nullptr;
    map = m;
    return true;
}

uint64_t block_session_bytes(const Geometry& g, const BlockDrafter& bd) {
    if (!bd.present) return 0;
    return bd.layers.size() * 2 * (uint64_t) (bd.window + 256) * bd.n_kv * bd.head_dim * 2 + (uint64_t) kWindow * (g.n_embd * 4 + 16) +
           64 * 256;
}

Model::~Model() {
    for (BatchGraph& g : batch_graphs) cudaGraphExecDestroy(g.exec);
    if (gate_side) cudaStreamDestroy(gate_side);
    if (gate_fork) cudaEventDestroy(gate_fork);
    if (gate_join) cudaEventDestroy(gate_join);
    if (stage_side) cudaStreamDestroy(stage_side);
    if (stage_free) cudaEventDestroy(stage_free);
    if (stage_ready) cudaEventDestroy(stage_ready);
    for (void* p : owned) cudaFree(p);
    vision_detach();
}

bool Model::vision_attach(int tokens, std::string& err) {
    if (!encoder) {
        err = o.arch + ": started without --vision";
        return false;
    }
    const size_t need = encoder->arena_bytes(tokens);
    if (enc_arena && enc_bytes >= need) return true;
    vision_detach();
    if (cudaMalloc(&enc_arena, need) != cudaSuccess) {
        cudaGetLastError();
        enc_arena = nullptr;
        err = o.arch + ": no device memory for the image encoder (" + std::to_string((unsigned long long) (need >> 20)) + " MiB)";
        return false;
    }
    enc_bytes = need;
    if (!encoder->attach(enc_arena, enc_bytes, tokens, err)) {
        vision_detach();
        return false;
    }
    return true;
}

bool Model::vision_cpu_ready(std::string& err) {
    if (cpu_encoder) return true;
    if (o.mmproj.empty()) {
        err = o.arch + ": no vision/mmproj section for the CPU encoder";
        return false;
    }
    auto mp = std::make_unique<vision::Mmproj>();
    auto enc = std::make_unique<vision::CpuEncoder>();
    const int threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);   // as halocline-vision's default
    if (!mp->open(o.mmproj, err) || !enc->init(*mp, threads, true, false, err)) {
        err = o.arch + ": the CPU image encoder: " + err;
        return false;
    }
    cpu_mmproj = std::move(mp);
    cpu_encoder = std::move(enc);
    return true;
}

void Model::vision_detach() {
    if (!enc_arena) return;
    if (encoder) encoder->detach();
    cudaFree(enc_arena);
    enc_arena = nullptr;
    enc_bytes = 0;
}

bool Model::ready_products(std::string& err) {
    mmq = std::make_unique<prefill::mmq::Context>();
    std::vector<const Layer*> all;
    for (const Layer& L : w.layers) all.push_back(&L);
    if (o.drafter == Drafter::Mtp) all.push_back(&w.mtp.layer);
    std::vector<const Linear*> ls = {&w.head};
    if (w.draft_head.rows > 0) ls.push_back(&w.draft_head);   // --draft-vocab: its rows' products are registered too
    if (o.drafter == Drafter::Mtp) ls.push_back(&w.mtp.fc);
    if (o.drafter == Drafter::Block) {
        for (const Linear& l : w.dflash.fc) ls.push_back(&l);
        for (const BlockDrafter::Layer& D : w.dflash.layers)
            for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) ls.push_back(l);
    }
    for (const Layer* Lp : all)
        for (const Linear* l : {&Lp->qkv, &Lp->z, &Lp->out, &Lp->q, &Lp->k, &Lp->v, &Lp->o, &Lp->gate, &Lp->up, &Lp->down})
            if (l->w) ls.push_back(l);
    for (const Linear* l : ls) {
        if (l->type == kTypeF8 || l->type == kTypeNvfp4Planes) continue;   // the window gemm's, not MMQ's
        if (!prefill::mmq::fits(l->type, l->rows)) {
            err = o.arch + ": the products have no tile for a " + std::to_string(l->rows) + "-row matrix";
            return false;
        }
        prefill::mmq::load(l->type, l->rows);
    }
    return true;
}

bool Model::ensure_chunk(std::string& err) {
    if (chunk_ready) return true;
    const Geometry& gg = g;
    if (!pre_arena.reserve(chunk_bytes(gg, paths.has_fp4, o.drafter))) {
        err = o.arch + ": out of device memory for the prompt workspaces";
        return false;
    }
    const size_t T = kChunk, H = (size_t) gg.n_embd, AQ = (size_t) gg.n_head * gg.head_dim, C = (size_t) gg.conv_channels();
    const size_t VD = (size_t) gg.value_width(), HV = (size_t) gg.gdn_v_heads, KV = (size_t) gg.n_kv * gg.head_dim;
    const size_t widest = std::max<size_t>({H, (size_t) gg.n_ff, VD, AQ});
    ChunkScratch& b = cs;
    b.x = pre_arena.take<float>(T * H);
    b.xn = pre_arena.take<float>(T * H);
    b.tmp = pre_arena.take<float>(T * H);
    b.xq = pre_arena.take_bytes(std::max(prefill::mmq::q8_bytes(T, (int64_t) widest), prefill::fp8::act_bytes(T, (int64_t) widest)));
    b.tok = pre_arena.take<int32_t>(T);
    b.pos = pre_arena.take<int32_t>(3 * T);
    b.iota = pre_arena.take<int32_t>(T);
    b.bounds = pre_arena.take<int32_t>(2);
    if (o.drafter == Drafter::Mtp) {
        b.cat = pre_arena.take<float>(T * H);
        b.hn = pre_arena.take<float>((T + 1) * H);
    }
    if (o.drafter == Drafter::Block) b.dacc = pre_arena.take<float>(T * H);
    // the FP4 activations (fp4_dense) of the widest product input and their token scales: taken when a product runs as
    // NVFP4 planes, which the chunks read through them; hmax stays for every pack (the gated rows' largest |value| is
    // written each attention chunk)
    if (paths.has_fp4) {
        b.a4 = pre_arena.take_bytes(prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) widest));
        b.gs = pre_arena.take<float>(T);
    }
    b.hmax = pre_arena.take<float>(T);
    const size_t phase = pre_arena.mark();
    if (gg.gdn_layers()) {
        b.qkv = pre_arena.take<float>(T * C);
        b.hc = pre_arena.take<float>(T * C);
        b.z = pre_arena.take<float>(T * VD);
        b.y = pre_arena.take<float>(T * VD);
        b.yn = pre_arena.take<float>(T * VD);
        b.gate = pre_arena.take<float>(T * HV);
        b.beta = pre_arena.take<float>(T * HV);
        b.ab = pre_arena.take<float>(T * 2 * HV);
        b.work = pre_arena.take_bytes(kernels::gdn_work_bytes(T));
        pre_arena.rewind(phase);
    }
    if (gg.attn_layers()) {
        b.qfull = pre_arena.take<float>(T * 2 * AQ);
        b.k = pre_arena.take<float>(T * KV);
        b.v = pre_arena.take<float>(T * KV);
        b.qo = pre_arena.take<float>(T * AQ);
        b.ao = pre_arena.take<float>(T * AQ);
        b.ag = pre_arena.take<float>(T * AQ);
        b.attn_split = pre_arena.take<float>(kernels::gqa_prompt_split_floats((int64_t) T, gg.n_head, kAttnSplits));
        pre_arena.rewind(phase);
    }
    b.g = pre_arena.take<float>(T * gg.n_ff);
    b.u = pre_arena.take<float>(T * gg.n_ff);
    b.h = pre_arena.take<float>(T * gg.n_ff);
    if (!b.h || !b.x) {
        err = o.arch + ": the prompt workspaces do not fit their arena";
        return false;
    }
    prefill::mmq::iota(b.iota, (int64_t) T, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = o.arch + ": preparing the prompt kernels failed";
        return false;
    }
    chunk_ready = true;
    return true;
}

ModelPtr make_model(const Geometry& g, const Weights& w, const Options& o, std::string& err) {
    ModelPtr m(new Model());
    m->g = g;
    m->w = w;
    m->o = o;
    m->paths = scratch_paths(w);
    if (const std::string e = g.check(); !e.empty()) {
        err = o.arch + ": " + e;
        return nullptr;
    }
    for (int il = 0; il < g.n_layers; ++il) {
        const Mixer k = g.kinds[(size_t) il];
        m->gdn_index.push_back(k == Mixer::Gdn ? m->n_gdn++ : -1);
        m->attn_index.push_back(k == Mixer::Attention ? m->n_attn++ : -1);
    }
    if (o.vision) {   // the projector stays in page-locked RAM; each request's pictures take it to the card
        m->mmproj = std::make_unique<vision::Mmproj>();
        m->encoder = std::make_unique<vision::CudaEncoder>();
        if (o.mmproj.empty()) {
            err = o.arch + ": --vision needs the pack's vision/mmproj section";
            return nullptr;
        }
        if (!m->mmproj->open(o.mmproj, err) || !m->encoder->stage(*m->mmproj, true, err) || !m->encoder->create(err)) {
            err = o.arch + ": vision: " + err;
            return nullptr;
        }
        if (m->mmproj->hp().proj != g.n_embd) {
            err = o.arch + ": the vision projector's width differs from the model's";
            return nullptr;
        }
        m->mmproj->release();
        halo::log::info("halo dense", "image encoder: %.2f GiB staged in page-locked RAM, on the card per request",
                          (double) m->encoder->weights_bytes() / 1073741824.0);
    }
    // the 4-bit caches have no streamed form, and the stage below is carved in fp16's rows: refuse both here, not only at
    // the flags, so a forced option cannot carve a layout the kernels never read
    if ((o.kv == KvFormat::K8V4 || o.kv == KvFormat::Q4) && o.kv_resident > 0 && o.kv_resident < o.max_context) {
        err = o.arch + ": " + (o.kv == KvFormat::Q4 ? "q4_0" : "k8v4") + " has no streamed form (--kv-resident)";
        return nullptr;
    }
    if (!m->dec_arena.reserve(window_bytes(g, o.drafter == Drafter::Block ? w.dflash : BlockDrafter{}, o.max_context,
                                          m->paths.has_wg))) {
        err = o.arch + ": out of device memory for the decode scratch";
        return nullptr;
    }
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, KV = (size_t) g.n_kv * g.head_dim;
    WindowScratch& b = m->ws;
    Arena& a = m->dec_arena;
    b.x = a.take<uint16_t>(W * H);
    b.xn = a.take<float>(W * H);
    b.tmp = a.take<float>(W * H);
    b.dacc = a.take<float>(W * H);
    if (m->paths.has_wg || o.drafter == Drafter::Block) {
        b.xq = a.take_bytes(act_slot(g));
        b.hq = a.take_bytes(act_slot(g));
        wgemm_need(g, b.w8.partial_floats, b.w8.tickets_n);
        b.w8.partial = a.take<float>(std::max<size_t>(b.w8.partial_floats, 1));
        b.w8.tickets = a.take<int32_t>(std::max<size_t>(b.w8.tickets_n, 1));   // zero: the arena is cleared below
    }
    b.x8 = a.take_bytes(q8_slot(g));
    b.z = a.take<float>(W * g.value_width());
    b.qfull = a.take<float>(W * 2 * AQ);
    b.k = a.take<float>(W * KV);
    b.v = a.take<float>(W * KV);
    b.qo = a.take<float>(W * AQ);
    b.ao = a.take<float>(W * AQ);
    b.ag = a.take<float>(W * AQ);
    b.g = a.take<float>(W * g.n_ff);
    b.u = a.take<float>(W * g.n_ff);
    b.h = a.take<float>(W * g.n_ff);
    if (g.attn_layers() > 0) b.attn = a.take<float>(kernels::gqa_decode_scratch_floats(o.max_context, kWindow, g.n_head));
    b.tok = a.take<int32_t>(W);
    b.pos = a.take<int32_t>(W * 3);
    b.cat = a.take<float>(W * 2 * H);
    b.lp_scratch = a.take<float>(kernels::row_logprobs_scratch_floats(kWindow, g.n_vocab));
    if (o.drafter == Drafter::Block) {
        const BlockDrafter& bd = w.dflash;
        const size_t AQd = (size_t) bd.n_head * bd.head_dim, KVd = (size_t) bd.n_kv * bd.head_dim;
        b.bq = a.take<float>(W * AQd);
        b.bk = a.take<float>(W * KVd);
        b.bv = a.take<float>(W * KVd);
        b.bqo = a.take<float>(W * AQd);
        b.bao = a.take<float>(W * AQd);
        b.dyn = a.take<float>(W * bd.layers[0].conv_a.rows);
        b.bh = a.take<float>(W * H);
        b.blp = a.take<float>(W * kernels::kLogprobStride);
        b.bsel = a.take<float>(W * bd.rank);
        b.bpos = a.take<int32_t>(W * 3);
        b.battn = a.take<float>(kernels::gqa_decode_scratch_floats(bd.window + 256, kWindow, bd.n_head, bd.head_dim));
    }
    b.y = a.take<float>(W * g.value_width());
    b.hl = a.take<float>(W * (size_t) g.n_vocab);
    b.iota = a.take<int32_t>(W);
    b.bounds = a.take<int32_t>(2);
    b.pos0 = a.take<int32_t>(2);
    if (g.gdn_layers() > 0) {
        b.rq = a.take<float>(W * (size_t) g.conv_channels());
        b.gab = a.take<float>(2 * W * (size_t) g.gdn_v_heads);
        if (cudaStreamCreateWithFlags(&m->gate_side, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->gate_fork, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->gate_join, cudaEventDisableTiming) != cudaSuccess) {
            err = o.arch + ": creating the GDN gate stream failed";
            return nullptr;
        }
    }
    b.btab = a.take<BatchTable>(1);
    if (!b.bounds) {
        err = o.arch + ": the decode scratch does not fit its arena";
        return nullptr;
    }
    // rows past a window's last stay finite: they are quantized with it
    if (cudaMemset(b.x, 0, a.mark()) != cudaSuccess) {   // b.x is the arena's first buffer
        err = o.arch + ": clearing the decode scratch failed";
        return nullptr;
    }
    prefill::mmq::iota(b.iota, kBatch, nullptr);
    if (const int64_t kept = resident_cells(o.max_context, o.kv_resident); kept < o.max_context && g.attn_layers() > 0) {
        // the prompt chunks' VRAM copy of one layer's far cells, laid out as a session's far buffers
        const size_t cells = (size_t) (o.max_context - kept) * g.n_kv * g.head_dim;
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) (o.max_context - kept) * kv_cell_bytes(g, o.kv)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&m->stage_side, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->stage_free, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->stage_ready, cudaEventDisableTiming) != cudaSuccess) {
            if (p) cudaFree(p);
            err = o.arch + ": out of device memory for the streamed K/V's prompt copy";
            return nullptr;
        }
        m->owned.push_back(p);
        kernels::KvFar& f = m->stage;
        f.at = kept;
        if (o.kv == KvFormat::Int8) {
            f.q8.k = (int8_t*) p;
            f.q8.v = f.q8.k + cells;
            f.q8.ks = (uint16_t*) (f.q8.v + cells);
            f.q8.vs = f.q8.ks + cells / 64;
        } else {
            f.k = (uint16_t*) p;
            f.v = f.k + cells;
        }
    }
    if (!decode_drafter(*m, err)) return nullptr;
    if (!m->ready_products(err) || cudaDeviceSynchronize() != cudaSuccess) {
        if (err.empty()) err = o.arch + ": readying the products failed";
        return nullptr;
    }
    return m;
}

void destroy(Model* m) { delete m; }

}  // namespace halo::dense
