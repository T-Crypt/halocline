"""The sampling-distribution gate (docs/roadmap.md P0): seeded runs against an OpenAI-compatible chat endpoint (`halo
serve`, llama.cpp's llama-server), then the frequencies of the sampled tokens compared with a reference build's
run-to-run interval.

    sampling_dist.py run --url http://127.0.0.1:8091 --model NAME --prompts prompts.jsonl --seeds 0-31 --out ref-a.jsonl
    sampling_dist.py run ... --seeds 32-63 --out ref-b.jsonl          # the reference build again, other seeds
    sampling_dist.py run ... --seeds 0-31 --out cand.jsonl            # the candidate build, the first run's seeds
    sampling_dist.py compare ref-a.jsonl ref-b.jsonl cand.jsonl

Per run, every sampled position gives: was the pick the row's top-1 (`top1`), was it among the row's 10 likeliest
(`top10`), and its log-probability (`logprob`), all from the endpoint's own `top_logprobs`. Per prompt, the first
picked token across seeds is a sample of one distribution; `first_tv` is the total-variation distance between two
runs' first-token histograms.

`compare` takes, per prompt, the candidate's metric minus reference A's and checks it against the reference build's
own spread: B minus A (other seeds), resampled over prompts (bootstrap, 95%). The candidate passes a metric when its
mean difference lies inside that interval. The candidate shares A's seeds, so an identical build differs by 0.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

METRICS = ("top1", "top10", "logprob")


def seeds_of(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_prompts(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "messages" not in r:
                r["messages"] = [{"role": "user", "content": r["prompt"]}]
            rows.append(r)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: prompt ids repeat")
    return rows


def request_body(a, prompt: dict, seed: int) -> dict:
    body = {"model": a.model, "messages": prompt["messages"], "seed": seed, "temperature": a.temperature, "top_p": a.top_p,
            "top_k": a.top_k, "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 10, "stream": False}
    body.update(json.loads(a.extra) if a.extra else {})
    return body


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def picks(resp: dict) -> list[dict]:
    """Each sampled position as {token, logprob, top1, top10}, from choices[0].logprobs.content."""
    content = ((resp.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
    out = []
    for e in content:
        alts = [x["token"] for x in e.get("top_logprobs") or []]
        out.append({"token": e["token"], "logprob": e["logprob"], "top1": bool(alts) and alts[0] == e["token"],
                    "top10": e["token"] in alts[:10]})
    return out


def run(a) -> int:
    prompts = load_prompts(a.prompts)
    seeds = seeds_of(a.seeds)
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"url": a.url, "model": a.model, "seeds": seeds, "temperature": a.temperature,
                                     "top_p": a.top_p, "top_k": a.top_k, "max_tokens": a.max_tokens, "extra": a.extra,
                                     "prompts": str(a.prompts), "label": a.label}}) + "\n")
        for p in prompts:
            for s in seeds:
                ps = picks(post(a.url, request_body(a, p, s), a.timeout))
                if not ps:
                    print(f"run: prompt {p['id']} seed {s}: no logprobs in the reply", file=sys.stderr)
                    return 1
                f.write(json.dumps({"id": p["id"], "seed": s, "picks": ps}) + "\n")
            print(f"  {p['id']}: {len(seeds)} seeds, {time.time() - t0:.0f} s", flush=True)
    return 0


def load_run(path) -> tuple[dict, dict]:
    """(meta, {prompt id: [records]})."""
    meta, by = {}, {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "meta" in r:
            meta = r["meta"]
        else:
            by.setdefault(r["id"], []).append(r)
    return meta, by


def per_prompt(by: dict) -> dict:
    """{prompt id: {metric: mean over its seeds' positions, "first": Counter of first tokens}}."""
    out = {}
    for pid, recs in by.items():
        allp = [x for r in recs for x in r["picks"]]
        m = {k: sum(float(x[k]) for x in allp) / len(allp) for k in METRICS}
        m["first"] = Counter(r["picks"][0]["token"] for r in recs)
        out[pid] = m
    return out


def tv(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def interval(diffs: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    """The 95% bootstrap interval of the mean of `diffs` (resampling prompts)."""
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def compare(ref_a, ref_b, cand, n_boot: int = 2000, seed: int = 0) -> dict:
    """The comparison table: per metric the candidate's mean difference to A, the reference spread's interval and the
    verdict. `first_tv` passes when TV(candidate, A) is at most the upper end of TV(B, A)'s interval."""
    A, B, C = (per_prompt(load_run(p)[1]) for p in (ref_a, ref_b, cand))
    ids = sorted(set(A) & set(B) & set(C))
    if not ids or len(ids) < len(A):
        raise ValueError(f"the runs share {len(ids)} of {len(A)} prompts")
    rng = random.Random(seed)
    rows = {}
    for k in METRICS:
        d_ref = [B[i][k] - A[i][k] for i in ids]
        d_c = sum(C[i][k] - A[i][k] for i in ids) / len(ids)
        lo, hi = interval(d_ref, n_boot, rng)
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        rows[k] = {"cand_minus_a": d_c, "ref_interval": [lo, hi], "a": sum(A[i][k] for i in ids) / len(ids),
                   "pass": lo <= d_c <= hi}
    t_ref = [tv(B[i]["first"], A[i]["first"]) for i in ids]
    t_c = sum(tv(C[i]["first"], A[i]["first"]) for i in ids) / len(ids)
    lo, hi = interval(t_ref, n_boot, rng)
    rows["first_tv"] = {"cand_vs_a": t_c, "ref_interval": [lo, hi], "pass": t_c <= hi}
    rows["prompts"] = len(ids)
    rows["pass"] = all(rows[k]["pass"] for k in (*METRICS, "first_tv"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="seeded requests, one JSON line per prompt and seed")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.jsonl")))
    p.add_argument("--seeds", default="0-31")
    p.add_argument("--temperature", type=float, default=1.0)   # Qwen3.8's sampling defaults (general.sampling.*)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--extra", help='JSON merged into every request, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    p.add_argument("--label", default="")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare", help="reference A, reference B (other seeds), candidate (A's seeds)")
    p.add_argument("ref_a")
    p.add_argument("ref_b")
    p.add_argument("cand")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    rows = compare(a.ref_a, a.ref_b, a.cand)
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        print(f"{'metric':<10} {'A':>9} {'cand - A':>10} {'reference B - A, 95%':>24}  verdict   ({rows['prompts']} prompts)")
        for k in METRICS:
            r = rows[k]
            print(f"{k:<10} {r['a']:>9.4f} {r['cand_minus_a']:>+10.4f} [{r['ref_interval'][0]:+.4f}, {r['ref_interval'][1]:+.4f}]"
                  f"  {'pass' if r['pass'] else 'FAIL'}")
        r = rows["first_tv"]
        print(f"{'first_tv':<10} {'':>9} {r['cand_vs_a']:>10.4f} [{r['ref_interval'][0]:.4f}, {r['ref_interval'][1]:.4f}]"
              f"  {'pass' if r['pass'] else 'FAIL'}")
        print("PASS" if rows["pass"] else "FAIL")
    return 0 if rows["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
