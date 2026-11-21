#!/usr/bin/env python3
"""Image time to first token, per image, on one Halocline server (2026-11-20, --vision cpu vs dynamic).

    python3 vision_timing.py MODEL --url http://127.0.0.1:8096 --images DIR --prompts FILE.json --out OUT.json

One fresh request per image (no shared prefix), greedy, thinking off, streamed, max_tokens 64. The prompt is the
challenge text with the URL sentence replaced by "Look at this image." Records time to first token (the encode plus
the prefill), the whole wall time, the prompt tokens and the reply, so two arms can be compared reply by reply.
After the images, two text-only decode probes (400 tokens) show whether decode speed is back after an image.
The images and prompts (airbench vision set, fetched by the 2026-11-19 dense run) stay outside the repo.
"""
import argparse, base64, json, re, sys, time, urllib.request


def stream(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; parts = []; usage = {}; n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if first is None:
                        first = time.time()
                    parts.append(piece); n += 1
    end = time.time()
    return {"ttft_s": round((first or end) - t0, 3), "wall_s": round(end - t0, 3), "reply": "".join(parts).strip(),
            "usage": usage, "chunks": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--url", required=True)
    ap.add_argument("--images", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chal = json.load(open(a.prompts))["challenges"]
    res = []
    for c in chal:
        key = c["key"]
        q = re.sub(r"Look at the image at \S+ \(fetch it and view it\)\.", "Look at this image.", c["prompt"])
        b64 = base64.b64encode(open(f"{a.images}/{key}.png", "rb").read()).decode()
        body = {"model": a.model, "temperature": 0, "max_tokens": 64, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}
        try:
            r = stream(a.url, body)
        except Exception as e:  # report, keep going
            r = {"error": str(e)}
        r["key"] = key
        res.append(r)
        print(f"{key:20s} ttft {r.get('ttft_s', 0):7.2f} s  wall {r.get('wall_s', 0):7.2f} s  "
              f"prompt={r.get('usage', {}).get('prompt_tokens')}  {r.get('reply', r.get('error', ''))[:60]!r}", flush=True)
    probes = []
    for i in range(2):
        body = {"model": a.model, "temperature": 0, "max_tokens": 400, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": f"Write a plain account, about 400 words, of how a lighthouse keeper's day ran in 1900 (version {i})."}]}
        r = stream(a.url, body)
        toks = r["usage"].get("completion_tokens") or r["chunks"]
        r["decode_tps"] = round(toks / max(1e-6, r["wall_s"] - r["ttft_s"]), 1)
        probes.append(r)
        print(f"text probe {i}: {toks} tok, decode {r['decode_tps']} t/s, ttft {r['ttft_s']} s", flush=True)
    t = [r["ttft_s"] for r in res if "ttft_s" in r]
    summary = {"images": len(t), "ttft_sum_s": round(sum(t), 1), "ttft_median_s": sorted(t)[len(t) // 2] if t else None,
               "errors": sum(1 for r in res if "error" in r)}
    print("summary", json.dumps(summary), flush=True)
    json.dump({"model": a.model, "utc": time.strftime("%FT%TZ", time.gmtime()), "summary": summary,
               "results": res, "probes": probes}, open(a.out, "w"), indent=1)
    sys.exit(1 if summary["errors"] else 0)


if __name__ == "__main__":
    main()
