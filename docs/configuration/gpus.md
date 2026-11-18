# GPUs

Halocline runs on one NVIDIA Ada card, compute capability 8.9 (sm_89). The build compiles device code for 8.9
alone, and the start refuses a visible card of any other architecture
([SPECS.md](../../SPECS.md)). The multi-card paths came from Basalt v0.4.1 (`d3b75742`) and were removed in
`575a92c0`: there is no expert tier, layer split, prompt split or tensor parallel, and no second card is used in any
mode. `33cfca03` made sm_89 the only target.

One setting picks the card: `gpu`. It means the same in the server (`--gpu`, or `"gpu"` in a model entry), in
`halocline` itself, in `halocline-bench` and in `halocline-probe`.

| Value | Card used |
| --- | --- |
| not set | GPU 0; one warning at start names the other visible sm_89 cards |
| `"0"`, `"1"`, ... | exactly that card |
| `"auto"` | the sm_89 card with the most memory bandwidth, then the most VRAM, then the lowest number; a card that is not sm_89 is skipped with one line naming it |

Cards are numbered as `nvidia-smi` numbers them. When `CUDA_VISIBLE_DEVICES` is set, the numbers count within its list
(`CUDA_VISIBLE_DEVICES=1,0` makes `gpu 0` the card `nvidia-smi` calls 1), and its entries are read as `nvidia-smi`
numbers or UUID prefixes. A number past the visible cards, a card listed twice or a card that is not sm_89 stops the
start with an error.

A list of two or more cards is refused: `Halocline runs on exactly one GPU; N were chosen`. The server asks the
engine for the choice before anything loads, prints its notes and warning, and gives the engine only that card,
renumbered from 0.

## What the card holds

The one card holds the weights, the KV cache and the drafter. On Flash-Next it also holds the expert cache: the
experts that do not fit there stay in page-locked RAM and stream over PCIe. On Qwen3.8-27B every layer is on the
card and only the token embedding waits in page-locked RAM.

## Checking the choice

`--gpu-plan` prints the choice as one JSON line and exits, without loading a model: `devices` (the
`CUDA_VISIBLE_DEVICES` spelling), `cards` (visible index, token, nvidia-smi number and name), `notes`, `warning` and
`error`. A model file's flags decide which cards a model may use, so name the model:

```sh
bin/halocline --model PACK.halo --gpu auto --gpu-plan
```

The server prints the same notes and warning at start. `--vision <GPU number>` runs the image encoder on that card;
with one card that is card 0 ([image input](../running.md#vision---vision)).
