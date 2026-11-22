// src/models/inkling2/vision.hpp - Inkling 2 Mini's image encoder (vision.cu).
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace halo {
class GgufFile;
}

namespace halo::models::inkling2 {

class VisionEncoder {
public:
    ~VisionEncoder();
    /// From the pack's vision/mmproj (projector inkling2v).
    bool load(const GgufFile& mmproj, std::string& err);
    /// A JPEG/PNG/BMP/GIF picture to `n_rows` embedding rows of the text width (the merged grid gx x gy, row-major).
    bool encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, int& gx, int& gy, std::string& err,
                void* stream);
    int max_tokens_ = 1024;   ///< merged tokens a picture may take (a larger one shrinks with its ratio kept)

private:
    int D_ = 1024, ff_ = 4096, layers_ = 23, heads_ = 16, patch_ = 20, merge_ = 2, out_ = 2304, grid_ = 0;
    float eps_ = 1e-5f;
    void* host_ = nullptr;
    std::map<std::string, const uint16_t*> hostw_;
    std::map<std::string, const float*> devw_;
    std::vector<void*> dev_;
    std::vector<float> pos_grid_;
};

}  // namespace halo::models::inkling2
