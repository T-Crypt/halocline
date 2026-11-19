// halo/vision/mmproj.hpp - the vision projector file (GGUF, architecture "clip"): hyper-parameters and tensors.
//
// Opened as a read-only mapping of the file, so the tensor bytes are file-backed pages: they never become
// anonymous memory of the reader, the page cache can reclaim them under memory pressure and MemAvailable counts
// them (the CPU encoder sidecar reads them in place this way).  Only the layout the Qwen3-VL style encoder uses is
// checked, and a file that does not match is refused with the reason.
#pragma once

#include "halo/core/model_file.hpp"
#include "halo/platform/mapped_file.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace halo::vision {

struct TensorRef {
    std::string name;
    uint32_t type = 0;               ///< ggml type id: 0 F32, 1 F16, 30 BF16
    std::vector<int64_t> ne;         ///< ggml order: ne[0] is the contiguous axis
    const void* data = nullptr;
    size_t bytes = 0;
    int64_t elems() const { int64_t n = 1; for (int64_t d : ne) n *= d; return n; }
};

struct Hparams {
    int image_size = 0, patch = 0, embd = 0, ffn = 0, layers = 0, heads = 0, merge = 0, proj = 0;
    int min_pixels = 0;   ///< smallest input area in pixels, when the file states one
    float eps = 0.f;
    float mean[3] = {0, 0, 0}, std[3] = {1, 1, 1};
};

class Mmproj {
public:
    Mmproj() = default;
    ~Mmproj();
    Mmproj(const Mmproj&) = delete;
    Mmproj& operator=(const Mmproj&) = delete;

    /// Reads the projector GGUF at `range` in place through a read-only mapping of the file (a whole file, or the
    /// `vision/mmproj` section of a .halo), parses the header and checks it is the `projector` type.  False with
    /// `err` otherwise.
    bool open(const core::FileRange& range, std::string& err, const char* projector = "qwen3vl_merger");

    /// The same parser over `bytes` already in memory (the pre-mapping load path, kept for the parity tests):
    /// `data` must outlive the Mmproj.  False with `err` otherwise.
    bool open_bytes(const uint8_t* data, size_t bytes, std::string& err, const char* projector = "qwen3vl_merger");

    /// Drops the file's bytes: tensor pointers are invalid afterwards, hp() stays.  For callers that copied the weights.
    void release();

    const Hparams& hp() const { return hp_; }
    /// A tensor by name; null when absent.
    const TensorRef* find(const std::string& name) const;
    /// A required tensor of the given type and element count (ne product); null with `err` set otherwise.
    const TensorRef* need(const std::string& name, uint32_t type, int64_t elems, std::string& err) const;

private:
    bool parse(const uint8_t* base, size_t bytes, std::string& err, const char* projector);
    platform::MappedFile mapping_;   ///< the file's view: the tensor data lives inside it
    Hparams hp_;
    std::map<std::string, TensorRef> tensors_;
};

}  // namespace halo::vision
