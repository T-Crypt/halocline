// src/program/prompt_chunks.cpp - see include/basalt/program/prompt_chunks.hpp.
#include "basalt/program/prompt_chunks.hpp"

#include <algorithm>

namespace basalt::program {

namespace {
int64_t grid_chunk(int64_t tokens, int64_t cap) {   // `tokens` up on the 256-token grid, at most `cap`
    if (tokens <= 0 || cap <= 0) return 0;
    return std::min(cap, (tokens + 255) / 256 * 256);
}
}  // namespace

std::pair<int64_t, int64_t> prompt_one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, int64_t head_chunk) {
    int64_t want = grid_chunk(tokens, cap);
    if (want > 0 && tokens > cap) {
        const int64_t parts = (tokens + cap - 1) / cap;
        want = grid_chunk((tokens + parts - 1) / parts, cap);
    }
    int64_t head = 0;
    if (head_ok && tokens > cap && tokens >= 2 * head_chunk && cap > head_chunk) {
        const int64_t rem = tokens - head_chunk, parts = (rem + cap - 1) / cap;
        want = grid_chunk((rem + parts - 1) / parts, cap);
        head = head_chunk;
    }
    if (head >= want) head = 0;
    return {want, head};
}

std::vector<int64_t> prompt_chunk_list(int64_t tokens, int64_t chunk, int64_t head) {
    std::vector<int64_t> v;
    const int64_t first = tokens > chunk && head > 0 ? std::min(head, chunk) : chunk;
    for (int64_t at = 0; chunk > 0 && at < tokens;) {
        const int64_t T = std::min(at == 0 ? first : chunk, tokens - at);
        v.push_back(T);
        at += T;
    }
    return v;
}

}  // namespace basalt::program
