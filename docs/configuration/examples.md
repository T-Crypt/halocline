# Example configurations

> [!WARNING]
> Carried over from Basalt's user docs with the names changed. Parts of this page describe code Halocline removed (multi-GPU, sm_120/Blackwell, Windows, DeepSeek) or packs it does not use (NVFP4). Halocline runs on one sm_89 card with Unsloth GGUF packs; [SPECS.md](../../SPECS.md) and [RESEARCH.md](../../RESEARCH.md) are current. This page is due for a rewrite.

Start with `bin/halocline serve --config halo_config.json`. Model paths are relative to the file. Settings: [configuration](index.md).

## One GPU (32 GB) and RAM

```json
{
  "version": 2,
  "models": [{ "model": "model.halo", "gpu": "0", "max-context": 262144, "kv": "int8" }],
  "engine": { "resident-experts": true, "vram-reserve-mib": 700 },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

## A GPU and a smaller second card (peer expert tier)

Like the first, with `"gpu": "0,1"` and `"vision": 1`, which runs the image encoder on the second card too.

## A low-VRAM card

Like the first, with `"max-context": 65536`, `"kv": "q4_0"` and `"vram-reserve-mib": 1024`.

Several models: [models and the router](models.md). `"relaxed-drafting": 0.5` in an entry trades exactness under sampling for speed
([precision](../precision/qwen38-flash-next.md#relaxed-drafting)).
