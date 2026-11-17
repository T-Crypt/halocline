#include "run.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/progress.hpp"
#include <string>
#include <future>

namespace basalt::prefill {
using namespace detail;

int Prefill::Run::Chunk::finish() {
    if (!ple_land()) return -1;   // a stage that ends before layer 1: the rows land anyway, the next gather starts
    if (issuer.joinable()) {
        issuer.join();
        owner.stats_.ms_experts_host += iss_ms;
        owner.stats_.experts_streamed += iss_streamed;
        owner.stats_.experts_dma += iss_dma;
    }
    owner.stats_.tokens += T;
    core::progress_at("reading the prompt (batched): finishing the chunk from token", p0);
    if (owner.on_chunk) {
        if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
            err = std::string("prefill: ") + cudaGetErrorString(cudaGetLastError());
            return -1;
        }
        if (!owner.on_chunk(m.R, T, p0, err)) return -1;
    }
    return 0;
}

}  // namespace basalt::prefill
