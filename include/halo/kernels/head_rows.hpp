// include/halo/kernels/head_rows.hpp - vocabulary head rows picked by an id list, on the GPU.
//
// A head over a token subset (the drafters' frequency-ranked heads, FR-Spec arXiv 2502.14856) is the head's own rows
// gathered as stored: a row of a GGUF-quantized head is whole blocks and moves byte for byte, so the draft head's
// logits are the full head's at those ids, bit for bit.  `map_ids` is the other half: the subset's row index of a
// pick goes back to the token id the next step embeds and the verify pass scores.
#pragma once

#include <cstdint>

namespace halo::kernels {

/// dst row i = src row ids[i] (row_bytes each, multiple of 16, 4 or 1), for n rows.  `ids` is a device id list.
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream);
/// ids[t] = table[ids[t]] for n entries (a subset index back to a token id), one block of up to 64 entries.
void map_ids(int32_t* ids, const int32_t* table, int n, void* stream);

}  // namespace halo::kernels
