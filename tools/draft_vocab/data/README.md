# The draft vocabulary of `--draft-vocab`

Three rankings of Qwen3.8-27B's 248,320 tokens, cut from the corpus below. The engine reads one file as little-endian
int32 token ids, strictly ascending; row r of the draft head is a copy of the head's row `ids[r]`, and the verify pass
still scores the full head, so a file can only move the acceptance rate.

| file | ids | bytes | corpus coverage |
|---|---|---|---|
| `qwen38-27b-wikitext103-src-n32768.bin` | 32,768 | 131,072 | 98.78% of the corpus's token occurrences |
| `qwen38-27b-wikitext103-src-n65536.bin` | 65,536 | 262,144 | 100.00% (the corpus writes 63,500 distinct ids) |
| `qwen38-27b-wikitext103-src-n131072.bin` | 131,072 | 524,288 | 100.00% (67,572 rows the corpus never writes, by id) |

Each `.json` beside it is the provenance: the pack and its source GGUF's fingerprint, the tokenizer (248,320 ids,
`qwen35` pre-tokenizer, the 33 control ids always included), each corpus with its file count, bytes and sha256, N, the
covered fraction, and the exact command.

## Corpus

- **wikitext-103 train, 40 MB**: the repository's own gate corpus, the file `tools/depth_kld` made for the depth KLD
  bases (`/data/models/halocline/kld-bases/depth/wikitext103-train-40MB.txt`, file sha256
  `b308aaeace2255d95149a24f5fb51671ba69c791e09cf908db849d5dedc28f3f`). Prose is what most served output is.
- **the tree's sources** (`src/`, `include/`, `tools/`, 7.4 MB): code and the JSON/markdown around it, because the
  structured workload writes tool-call JSON and edits. `tools/draft_vocab` is excluded so that editing this tool never
  shifts the ranking; the counted tree's commit and the per-corpus digests are in each `.json`.

The mix is 40 MB prose to 7.4 MB code (about 5:1 by bytes): the served mix is prose-heavy, and the code's contribution
is the punctuation, `{"`, `":`, `\n`, `,` and the common identifiers a tool call repeats. Nothing here is tuned per
request: one fixed ranking is used for every prompt.

## Regenerating

```sh
python3 tools/draft_vocab/make_draft_vocab.py \
    --pack /data/models/halocline/packs/Qwen3.8-27B-UD-Q4_K_XL.halo \
    --corpus /data/models/halocline/kld-bases/depth/wikitext103-train-40MB.txt \
    --corpus src --corpus include --corpus tools --exclude tools/draft_vocab \
    --n 65536 \
    --out tools/draft_vocab/data/qwen38-27b-wikitext103-src-n65536.bin \
    --provenance tools/draft_vocab/data/qwen38-27b-wikitext103-src-n65536.json
```

Needs `regex` (in `requirements.txt`) and `numpy` for the pack's tokenizer. The ranking is the N most frequent ids,
ties to the lower id, with the pack's `tokenizer.ggml.*_token_id` control tokens always in. The file's ids are sorted
ascending; the engine refuses any other order, a duplicate, an empty file or an id outside the vocabulary by name.
