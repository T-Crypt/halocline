// tests/vision/mapped_identity_test.cpp - the CPU encoder's two projector backings give bit-identical embeddings.
//
// The projector file is a synthetic GGUF written here with the real Qwen3-VL geometry but one layer and an 8-wide
// projection (the encoder checks the geometry, not the values), so the test needs no pack.  One Mmproj opens the
// file through its read-only mapping (what the sidecar uses); a second parses the same bytes held in memory (the
// old read-into-heap load path).  The two encoders encode the same tiny synthetic image and their embeddings must
// be bit-identical, with both the sidecar's flags (round_bf16, no f16 quirks) and the parity program's (f16 quirks).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "halo/vision/encoder_cpu.hpp"

namespace {

using namespace halo::vision;

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const auto n = b.size();
    b.resize(n + sizeof(T));
    std::memcpy(b.data() + n, &v, sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

// A deterministic value for tensor element `i`, so the file is the same on every run.
float sample(size_t i) { return (float) ((i * 2654435761ull) % 2000) / 2000.0f - 0.5f; }

struct Spec {
    std::string name;
    uint32_t type;
    std::vector<int64_t> ne;   // ggml order: ne[0] is the contiguous axis
};

// A GGUF v3 with the `qwen3vl_merger` projector's real shapes (E=1152, F=4304, 16 heads) but one layer and
// proj=8.  Returns the file's bytes.
std::vector<uint8_t> build_gguf() {
    std::vector<uint8_t> b;
    put<uint32_t>(b, 0x46554747u);   // "GGUF"
    put<uint32_t>(b, 3);
    const std::vector<Spec> tensors = {
        {"v.patch_embd.weight", 0, {768, 1152}},
        {"v.patch_embd.weight.1", 0, {768, 1152}},
        {"v.patch_embd.bias", 0, {1152}},
        {"v.position_embd.weight", 0, {2304, 1152}},
        {"v.post_ln.weight", 0, {1152}},
        {"v.post_ln.bias", 0, {1152}},
        {"mm.0.bias", 0, {4608}},
        {"mm.2.bias", 0, {8}},
        {"mm.0.weight", 30, {4608, 4608}},
        {"mm.2.weight", 30, {8, 4608}},
        {"v.blk.0.ln1.weight", 0, {1152}},
        {"v.blk.0.ln1.bias", 0, {1152}},
        {"v.blk.0.ln2.weight", 0, {1152}},
        {"v.blk.0.ln2.bias", 0, {1152}},
        {"v.blk.0.attn_qkv.bias", 0, {3456}},
        {"v.blk.0.attn_out.bias", 0, {1152}},
        {"v.blk.0.ffn_up.bias", 0, {4304}},
        {"v.blk.0.ffn_down.bias", 0, {1152}},
        {"v.blk.0.attn_qkv.weight", 30, {3456, 1152}},
        {"v.blk.0.attn_out.weight", 30, {1152, 1152}},
        {"v.blk.0.ffn_up.weight", 30, {4304, 1152}},
        {"v.blk.0.ffn_down.weight", 30, {1152, 4304}},
    };
    put<uint64_t>(b, tensors.size());
    const std::vector<std::pair<std::string, uint32_t>> num_kv = {
        {"clip.vision.image_size", 4}, {"clip.vision.patch_size", 4},
        {"clip.vision.embedding_length", 4}, {"clip.vision.feed_forward_length", 4},
        {"clip.vision.block_count", 4}, {"clip.vision.attention.head_count", 4},
        {"clip.vision.spatial_merge_size", 4}, {"clip.vision.projection_dim", 4},
        {"clip.vision.attention.layer_norm_epsilon", 6}, {"general.alignment", 4},
    };
    put<uint64_t>(b, 2 + num_kv.size() + 2);   // n_kv
    put_str(b, "general.architecture");
    put<uint32_t>(b, 8);
    put_str(b, "clip");
    put_str(b, "clip.projector_type");
    put<uint32_t>(b, 8);
    put_str(b, "qwen3vl_merger");
    const uint32_t vals[] = {32, 16, 1152, 4304, 1, 16, 2, 8, 0x358637BDu /* 1e-6 as f32 bits */, 32};
    for (size_t i = 0; i < num_kv.size(); ++i) {
        put_str(b, num_kv[i].first);
        put<uint32_t>(b, num_kv[i].second);
        put<uint32_t>(b, vals[i]);
    }
    for (const char* k : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        put_str(b, k);
        put<uint32_t>(b, 9);
        put<uint32_t>(b, 6);
        put<uint64_t>(b, 3);
        for (uint32_t i = 0; i < 3; ++i) put<float>(b, 0.5f);
    }
    uint64_t off = 0;
    for (const Spec& t : tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t) t.ne.size());
        for (int64_t d : t.ne) put<uint64_t>(b, (uint64_t) d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, off);
        int64_t n = 1;
        for (int64_t d : t.ne) n *= d;
        off += (uint64_t) n * (t.type == 0 ? 4 : 2);
    }
    b.resize((b.size() + 31) / 32 * 32);
    size_t i = 0;
    for (const Spec& t : tensors) {
        int64_t n = 1;
        for (int64_t d : t.ne) n *= d;
        for (int64_t j = 0; j < n; ++j, ++i) {
            const float v = sample(i);
            if (t.type == 0) put<float>(b, v);
            else put<uint16_t>(b, (uint16_t) (i * 2654435761ull >> 16));   // bf16 bits
        }
    }
    return b;
}

std::filesystem::path write_file(const std::vector<uint8_t>& b) {
    const auto n = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() / ("vision_mapped_test-" + std::to_string(n) + ".gguf");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), (std::streamsize) b.size());
    return path;
}

}  // namespace

int main() {
    std::printf("vision_mapped_identity_test\n");
    const std::vector<uint8_t> gguf = build_gguf();
    const auto path = write_file(gguf);

    std::string err;
    Mmproj mapped;
    const bool mapped_ok = mapped.open(halo::core::FileRange(path.string()), err);
    check(mapped_ok, "the mapped projector opens");
    if (!err.empty()) std::printf("  (%s)\n", err.c_str());

    // the old load path's content: the same bytes read into memory
    std::vector<uint8_t> raw(gguf.size());
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size());
    check((size_t) f.gcount() == raw.size(), "the file reads back whole");

    Mmproj heap;
    const bool heap_ok = heap.open_bytes(raw.data(), raw.size(), err);
    check(heap_ok, "the in-memory projector opens");
    if (!err.empty()) std::printf("  (%s)\n", err.c_str());

    if (mapped_ok && heap_ok) {
        const Hparams& a = mapped.hp(), &b = heap.hp();
        check(a.image_size == b.image_size && a.patch == b.patch && a.embd == b.embd && a.ffn == b.ffn &&
                  a.layers == b.layers && a.heads == b.heads && a.merge == b.merge && a.proj == b.proj,
              "both backings read the same hyper-parameters");
        check(mapped.find("v.patch_embd.weight")->bytes == heap.find("v.patch_embd.weight")->bytes,
              "both backings read the same tensor sizes");

        // a tiny synthetic image: 2x2 patches of 768 values, the smallest canvas the encoder takes
        std::vector<float> patches(4 * 768);
        for (size_t i = 0; i < patches.size(); ++i) patches[i] = sample(i);
        const Size canvas{32, 32};

        for (const bool quirks : {false, true}) {
            CpuEncoder ea, eb;
            const bool ok = ea.init(mapped, 2, true, quirks, err) && eb.init(heap, 2, true, quirks, err);
            check(ok, quirks ? "both encoders initialize (f16 quirks on)" : "both encoders initialize (sidecar flags)");
            if (!ok) { std::printf("  (%s)\n", err.c_str()); continue; }
            std::vector<float> oa, ob;
            const bool ran = ea.encode(patches, canvas, oa) && eb.encode(patches, canvas, ob);
            check(ran, "both encoders encode the image");
            if (!ran) continue;
            bool nonzero = false;
            for (float v : oa) nonzero |= v != 0.0f;
            check(nonzero && oa.size() == 8, "the embedding is real work, not a zero pass");
            check(oa.size() == ob.size() && std::memcmp(oa.data(), ob.data(), oa.size() * sizeof(float)) == 0,
                  quirks ? "mapped and in-memory embeddings are bit-identical (f16 quirks on)"
                         : "mapped and in-memory embeddings are bit-identical (sidecar flags)");
        }
    }

    {   // the parser refuses bytes that cannot hold the header
        std::string e2;
        Mmproj short_open;
        check(!short_open.open_bytes(raw.data(), 16, e2) && !e2.empty(), "a truncated projector is refused");
    }

    std::filesystem::remove(path);
    std::printf(g_fail ? "vision_mapped_identity_test: %d FAILED\n" : "vision_mapped_identity_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
