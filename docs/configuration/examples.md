# Example configurations

Start with `bin/halocline serve --config halo_config.json`. Model paths are relative to the file. Settings:
[configuration](index.md). The repository's own `halo_config.json` is the sparse example below.

## Flash-Next on the reference machine

```json
{
  "version": 2,
  "models": [{ "model": "QFN-UD-IQ4_XS.halo", "gpu": "0", "max-context": 262144, "kv": "int8" }],
  "engine": { "resident-experts": true, "spec": 8, "prefill": "auto" },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

`resident-experts` is the default once the pack names an expert profile; it is written here to show where it goes.

## The dense 27B, served profile

```json
{
  "version": 2,
  "models": [{ "model": "Qwen3.8-27B-UD-Q4_K_XL-vision.halo", "gpu": "0", "max-context": 175616,
               "kv": "k8v4", "mtp-kv": "int8", "spec": 8, "vision": "cpu" }],
  "server": { "host": "127.0.0.1", "port": 8093 }
}
```

`vision` needs a pack with the projector. The dense guide explains the context and the KV format
([Qwen3.8-27B](../models/qwen38-27b.md)).

## Both models, one server

```json
{
  "version": 2,
  "models": [
    { "alias": "Flash-Next", "model": "QFN-UD-IQ4_XS.halo", "gpu": "0", "kv": "int8", "max-context": 262144,
      "idle-sleep": 300 },
    { "alias": "Qwen3.8-27B", "model": "Qwen3.8-27B-UD-Q4_K_XL-vision.halo", "gpu": "0", "kv": "k8v4",
      "mtp-kv": "int8", "max-context": 175616, "vision": "cpu", "idle-sleep": 300 }
  ],
  "engine": { "spec": 8, "prefill": "auto" },
  "server": { "host": "127.0.0.1", "port": 8080 }
}
```

One model loads at a time; a request for the other swaps engines ([models and the router](models.md)). A setting in
`engine` reaches both models only where the model takes it, so `prefill` is left out for the dense path.

## A smaller Ada card

Lower the context and let the engine size the expert cache from the free VRAM:

```json
{ "version": 2,
  "models": [{ "model": "QFN-UD-IQ4_XS.halo", "gpu": "0", "max-context": 65536, "kv": "int8" }],
  "server": { "host": "127.0.0.1", "port": 8080 } }
```

Placement and speed change with the card; re-run the quality gate before trusting a number from this repo
([the reference machine](../../SPECS.md)).
