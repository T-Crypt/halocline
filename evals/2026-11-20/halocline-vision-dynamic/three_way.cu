// three_way.cu - is the CUDA encoder further from full precision than the served CPU encoder? (2026-11-20)
//
//   three_way <mmproj.gguf> <image>...
//
// For each image: R = CPU encoder with fp32 activations (round_bf16 off), the closest thing to exact we have;
// C = CPU encoder with bf16 GEMM inputs (round_bf16 on: what `--vision cpu` serves); G = the CUDA encoder
// (`--vision dynamic` / GPU: bf16 GEMM inputs, fp16 attention with fp32 softmax). Prints the final embeddings'
// relative L2 and the worst row cosine of C and G against R, and G against C.
// Build: see build.sh beside this file (links the release's own vision libraries; no change to Halocline).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "halo/core/model_file.hpp"
#include "halo/vision/encoder_cpu.hpp"
#include "halo/vision/encoder_cuda.hpp"
#include "halo/vision/preprocess.hpp"
using namespace halo::vision;

static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& r, int dim) {
    double d = 0, n = 0, worst = 2;
    int worst_row = -1;
    const int rows = (int) (r.size() / dim);
    for (int i = 0; i < rows; ++i) {
        double ab = 0, aa = 0, bb = 0;
        for (int j = 0; j < dim; ++j) {
            const double x = a[(size_t) i * dim + j], y = r[(size_t) i * dim + j];
            ab += x * y; aa += x * x; bb += y * y; d += (x - y) * (x - y); n += y * y;
        }
        const double c = ab / std::sqrt(std::max(aa * bb, 1e-300));
        if (c < worst) { worst = c; worst_row = i; }
    }
    std::printf("  %-8s rel L2 %.3e  min row cosine %.5f (row %d of %d)\n", what, std::sqrt(d / std::max(n, 1e-300)),
                worst, worst_row, rows);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: three_way <mmproj.gguf> <image>...\n"); return 2; }
    std::string err;
    Mmproj mp;
    if (!mp.open(halo::core::FileRange(argv[1]), err)) { std::printf("%s\n", err.c_str()); return 2; }
    CpuEncoder exact, served;
    if (!exact.init(mp, 0, false, false, err) || !served.init(mp, 0, true, false, err)) { std::printf("%s\n", err.c_str()); return 2; }
    const Geometry geo;
    CudaEncoder gpu;
    if (!gpu.init(mp, geo.max_tokens, err)) { std::printf("cuda init: %s\n", err.c_str()); return 2; }
    for (int i = 2; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        Image img;
        if (!decode_image(bytes.data(), bytes.size(), img, err)) { std::printf("%s: %s\n", argv[i], err.c_str()); continue; }
        const Size sz = fit_size(img.w, img.h, geo);
        std::vector<float> planar, patches, r, c, g;
        normalize_planar(to_canvas(img, sz), planar);
        patchify(planar, sz, geo, patches);
        if (!exact.encode(patches, sz, r) || !served.encode(patches, sz, c) || !gpu.encode(patches, sz, g)) { std::printf("%s: encode failed\n", argv[i]); continue; }
        const int tokens = sz.w / 32 * (sz.h / 32);
        const int dim = (int) (r.size() / tokens);
        std::printf("== %s (%d tokens of %d)\n", argv[i], tokens, dim);
        compare("cpu-bf16", c, r, dim);
        compare("gpu", g, r, dim);
        compare("gpu-cpu", g, c, dim);
        std::fflush(stdout);
    }
    return 0;
}
