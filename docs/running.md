# Running the server

Settings: [configuration](configuration/index.md). Diagnostic flags: `bin/halocline --help`. Several models: [models and the router](configuration/models.md).

## Start-up

The page is up at once and the model loads behind it. The header reads `Loading` with the percent over a progress bar
while weights load; the stats page's Load button for a model loads it. `/health` and `/metrics` show the load. A failed
start-up load ends the server. Requests wait during a load and get a 503 if it fails.

## Concurrent requests: `--parallel N`

`--parallel N` (1 to 8) serves N requests at once in batch slots. Further requests wait in arrival order. `--max-context` is shared:
each slot holds `max-context / N` tokens (`--parallel 4 --max-context 262144` gives four slots of 65,536; a slot never gets
under 256). A request longer than its slot is refused with a 400 that names the length. `--slot-full-context` gives every
slot the whole `--max-context`, at N times the KV cache.

A prompt is read on its own thread beside the decoding slots. A waiting request whose prompt is under half the running
read's length takes precedence: the read gives way at its next chunk boundary, at most twice, and continues in its
reserved slot.

### Session affinity

With `--parallel`, a request that names its session goes to the slot that session last used, so the next turn continues the K/V already
there. The session is the first of these headers that is set, then the request body's `session_id`: `x-llm-flow-id`,
`x-claude-code-session-id`, `x-session-id`, `x-session-affinity`. A request with none picks a slot by the longest shared prefix, then
the least recently used. When the session's slot is busy, the request takes that normal pick and the session's slot becomes the new
one. Claude Code sends one `x-claude-code-session-id` for a session and its parallel subagents, so their overlapping requests spread
over slots.

## The console log

Every line starts with `[server]`. Values are `key=value`. Start-up prints a legend, a line per model, `ready`, `loaded`, `kv`, `slots`.
While a prompt is read, a slot prints `pp=<read>/<total>` with the rate. While requests generate, a `live` line appears every 2 s.
Each request ends with one `done` line:

```
[server] s1 done  pp=96/89c 0.1s  tg=1720 119.6t/s acc=65% dl=2.9  ttft=0.1s  end=stop  hit=89.7%  pcie=39.3%
```

- `pp=<prompt tokens>/<cached>c`: the rate counts newly read tokens only.
- `acc`: drafts accepted over drafts offered. `dl`: tokens per window. Both are absent when the request did not draft.
- `end`: `stop`, `length`, `cancelled`, `disconnect` or `error`.
- `hit`: expert cache hit rate. `pcie`: share of routed experts read over PCIe. `ssd`: the share read from the SSD file,
  when some experts stay in it.

The engine's own `halo ...` lines go to `~/.cache/halo/<id>/engine.log`, one folder per model file (`<id>` is 16 hex digits
taken from the file's meta section); `[server] engine log: PATH` names it when the engine fails to start. The folder also holds the
conversation cache. Nothing is unpacked from the model file: the server reads the tokenizer from its sections, and the engine and `halocline-vision`
read the vision projector inside it.

## The stats page

The server's address serves a page, updated every second from `/metrics`, `/slots` and `/api/logs`: GPU load, power, VRAM and temperature,
RAM and disk, decode and prefill rates with 60 s charts, slots, the newest requests, and the console lines.

Three quiet figures sit in it, each from `/metrics`:

- `N chats cached · X GiB` at the right of the Speed footer: the conversations parked in RAM (`conversation`: `parked`,
  `bytes`, `evictions`, and the last restore's `restored` tokens and `restored_ms`, in the tooltip). The server reads them
  from the engine's `CONV` lines; a new engine process starts empty.
- `N requests served · M tokens written` under the Slots table (`totals`: requests, prompt, reused and written tokens, prompt
  and decode time since the server started; the rest in the tooltip).
- The header word reads `Thinking`, `Answering` or `Tool call` (`live.phase`) while exactly one request runs.

The log panel shows the engine's long cache lines short (`cache: restored 187k tokens in 912 ms`); the log and `/api/logs`
keep the raw line.

The stats page has an Engine tab and a Monitor page (2026-11-17), and a time on every request. The Engine tab reads `/metrics` and shows
the pack (name, path, quant, `max_context`), the KV format with its resident cells and elastic ceiling, the drafter (MTP or
block, verify window, cache format, window and ring) or `None`, the experts on the card, in the RAM arena and on the SSD
with the last request's hit and the PCIe share (a dense model shows its weights on the card and in RAM instead), the
conversation cache's budget and figures, and the machine's clocks, PCIe traffic, disk writes and driver. A stat the
loaded model does not have is left out, not shown as 0. The Speed footer names the engine's shape (`KV INT8 · MTP 6 ·
14.0k experts`). The Requests table gains time to first token and the wait before its turn; the endpoint moves to the
monitor's request detail. `/monitor`, linked from the header, is the request inspector: every kept request (the newest
12, or all 500 with `?requests=all`) with its prompt and output tokens, prefill and decode rates, wait, load, first
token, draft offers and acceptance, tokens per window, expert shares and the DONE timings. The server keeps no prompt or
answer text, so the detail's Content section says exactly that. `/metrics` also carries `engine.path`, `live.tok_s_mean`, `live.max_tokens`,
`requests_kept`, each GPU's `clock_sm_mhz` and `clock_mem_mhz`, `hardware.disk_write_mb`, and per request `queue_s`,
`load_s` and `timings`.

## Conversations on disk: `--conversation-cache-disk-mib`

Off at 0. With a budget, conversations of at least 8,192 tokens (4,096 on the dense path) are written to `conversations` in the model's cache folder when they leave
RAM or the session, or at exit. The image holds the conversation's KV state, its pictures and its checkpoints; `--prompt-cache`
sets how many checkpoints a conversation keeps. Each file is named for the model and every setting that changes the state's
bytes, and a read checks the headers and every block's hash: a mismatch drops the conversation. The folder keeps to its budget
(least recently used first). After a restart, a prompt that starts with a stored conversation reads it back.

## Sleep: `--idle-sleep`

After the given seconds without requests, the engine frees its VRAM (the driver moves the GPU state to host memory). The process and its
RAM copy of the experts stay; a dense model has no expert cache to give back first, so the driver moves all of its GPU state. The next
request wakes it in a few seconds. Linux only. The server stays awake when the GPU state would not fit in
free RAM with 8 GiB to spare. `--idle-unload` ends the process instead.

## Memory floors

Halocline refuses settings that would leave the machine without memory, and never shrinks what you asked for:

- `--vram-reserve-mib` below the floor is refused: 128 MiB on Linux, 256 MiB on WSL2, where the driver pages GPU memory.
- `--ram-reserve-mib` (default 2048) is the RAM left free for the rest of the system. The engine allocates its host buffers
  first, or holds their RAM when it page-locks them later; the expert tiers then take the free RAM less the reserve and less
  the driver's tracking of the pages they pin (`RAM floor ...` in the start log). The plan leaves 2 GiB beside the reserve
  for the host memory the engine takes after it has read the free RAM (CUDA graphs, the first request), so the free RAM
  settles above the guard, not on it. When some experts stay in the file, the
  tiers also leave a floor to the OS file cache they are read through: one prompt chunk's file reads and two layers' share of
  those experts, at least 4 GiB. Programs left idle beside a small reserve may be swapped out.
- While serving, the RAM guard holds the reserve: below it, it parks conversations to disk or drops them. A request waits up to 30 s
  for RAM to return. A running request stops with an error only when RAM is critical (under 1 GiB free, or below the reserve under
  memory pressure). `--ram-reserve-mib 0` turns the guard off.
- Experts left in the file show in the start log's `INFO spill_experts=... spill_bytes=...` line and in the stats page's
  Experts block (`N experts on SSD (X GiB)`), which turns amber.
- The disk tier's RAM floor is the reserve plus a GiB.
- `--resident-budget-gib` that would leave less than the floor is refused with the largest safe value.

## Vision: `--vision`

`--vision off | dynamic | cpu | <GPU number>` (config: `"vision"` in a model entry). The encoder is Halocline's own, and it
reads the projector in place from the model file's `vision/mmproj` section.

| Mode | Encoder runs | VRAM cost |
| --- | --- | --- |
| `off` | none | none |
| GPU number | sidecar `bin/halocline-vision` on that card | about 2.2 GiB while it runs |
| `dynamic` | inside the engine, on the main card | none at rest (the encoder waits in host RAM) |
| `cpu` | sidecar on the CPU | none, and slow |

Both packs carry the projector and both models serve images on the CPU encoder (`--vision cpu`); the engine repo's run is
[`halocline-vision`](../evals/2026-11-19/halocline-vision/). Which images a request may name:
[image sources](configuration/image-sources.md). An image may not sit in a system prompt.

## Video

With image input on, a user message names a video as it names an image, from the same
[sources](configuration/image-sources.md):

| API | Part |
| --- | --- |
| OpenAI chat | `{"type": "video_url", "video_url": {"url": "..."}}`; llama.cpp's `{"type": "input_video", "input_video": {"url": "..."}}` |
| Responses | `{"type": "input_video", "video_url": "..."}` |
| Anthropic | `{"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "..."}}`, or a `url` source |

Frames are sampled as the model's own processor samples them: 2 a second, 4 to 768, evenly spaced. A part may set `fps` (up to 60)
and `max_frames` (4 to 768). All frames are scaled to one size so the whole video fits `video-max-tokens` (default 8192 image
tokens), and each pair of frames is read as one picture with its time in the prompt. Only the video stream is read: audio is
ignored and rotation metadata is not applied, so a phone's portrait clip arrives sideways. A video may be 1 GiB and 10 minutes.

Decoding uses FFmpeg 6, 7, 8 or 9 from the machine ([platforms](platforms.md#ffmpeg-for-video)). At start the server prints
`[server] video on, FFmpeg 8 (...)`, or `off` with the reason, and then a video is refused with that reason.

Frames decode with NVDEC when the encoder runs on a card (`--vision <GPU number>`) and FFmpeg's build and the card allow it;
`--vision cpu` and `--vision dynamic` decode on the CPU. The encoder caches a video by its bytes and sampling, so a second
question about the same video reuses its embeddings.
