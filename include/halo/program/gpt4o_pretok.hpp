// include/halo/program/gpt4o_pretok.hpp - the GPT-4o (o200k) pre-tokenizer: the text split into the pieces BPE merges.
#pragma once

#include <cstddef>
#include <vector>

namespace halo::program::tokenizer {

/// The GPT-4o pre-tokenizer: `ends` gets the byte offset where each piece of the text ends.
void pretok_gpt4o(const char* text, size_t size, std::vector<size_t>& ends);

}  // namespace halo::program::tokenizer
