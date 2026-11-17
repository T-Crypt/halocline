// include/basalt/program/prompt_chunks.hpp - the chunks one card reads a prompt in.  Pure functions, no GPU calls.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::program {

/// How one card reads a part of `tokens` in chunks of at most `cap`: equal chunks on the 256-token grid for the fewest
/// chunks, and with `head_ok` a long part starts with a chunk of `head_chunk`.  The chunk, and the head chunk (0: none;
/// never one the chunk does not exceed, as the prompt path reads no first chunk longer than the rest).
std::pair<int64_t, int64_t> prompt_one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, int64_t head_chunk);

/// The sizes of the chunks the prompt path reads a part of `tokens` in, for a chunk and head chunk: a first chunk of
/// min(head, chunk) when the part is longer than one chunk, then chunks of `chunk`, the last one what is left.
std::vector<int64_t> prompt_chunk_list(int64_t tokens, int64_t chunk, int64_t head);

}  // namespace basalt::program
