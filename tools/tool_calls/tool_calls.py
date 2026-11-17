"""Tool-call breadth (docs/roadmap.md P0): labelled prompts that should or should not call a tool, run against an
OpenAI-compatible chat endpoint (`halo serve`, llama.cpp's llama-server), scored against the labels and against a
reference run.

    tool_calls.py run --url http://127.0.0.1:8091 --model NAME --out ref.jsonl [--cases cases.jsonl]
    tool_calls.py run ... --out cand.jsonl
    tool_calls.py score cand.jsonl [--ref ref.jsonl]

A case is {"id", "messages", "expect": "call" | "none", "tool": NAME, "args": [required keys]}; every case offers the
same tools (TOOLS below). A call case is right when the reply's first tool call names `tool` and its arguments parse as
a JSON object holding every key in `args`; a none case is right when the reply has no tool call. `score` prints the
accuracy per label and overall and, with --ref, how many cases the two runs decide alike (agreement) and which differ.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = [
    _fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"]),
    _fn("web_search", "Search the web and return the top results.", {"query": S}, ["query"]),
    _fn("calculator", "Evaluate an arithmetic expression exactly.", {"expression": S}, ["expression"]),
    _fn("read_file", "Read a text file from the user's project.", {"path": S}, ["path"]),
    _fn("send_email", "Send an email on the user's behalf.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    _fn("create_event", "Add an event to the user's calendar.",
        {"title": S, "start": {"type": "string", "description": "ISO 8601 date and time"}, "minutes": {"type": "integer"}},
        ["title", "start"]),
]


def load_cases(path) -> list[dict]:
    cases = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    names = {t["function"]["name"] for t in TOOLS}
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: case ids repeat")
    for c in cases:
        if c["expect"] not in ("call", "none"):
            raise ValueError(f"{c['id']}: expect is call or none")
        if c["expect"] == "call" and c.get("tool") not in names:
            raise ValueError(f"{c['id']}: tool {c.get('tool')!r} is not offered")
        if "messages" not in c:
            c["messages"] = [{"role": "user", "content": c["prompt"]}]
    return cases


def first_call(resp: dict):
    """(name, arguments string) of the reply's first tool call, or None."""
    msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    f = calls[0].get("function") or {}
    return f.get("name"), f.get("arguments") or ""


def judge(case: dict, call) -> dict:
    if call is None:
        return {"called": False, "right": case["expect"] == "none"}
    name, raw = call
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        args = None
    ok_args = isinstance(args, dict) and all(k in args for k in case.get("args", []))
    right = case["expect"] == "call" and name == case.get("tool") and ok_args
    return {"called": True, "name": name, "args_ok": ok_args, "right": right}


def run(a) -> int:
    cases = load_cases(a.cases)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"url": a.url, "model": a.model, "cases": str(a.cases), "label": a.label,
                                     "extra": a.extra}}) + "\n")
        for c in cases:
            body = {"model": a.model, "messages": c["messages"], "tools": TOOLS, "tool_choice": "auto", "temperature": 0,
                    "seed": 0, "max_tokens": a.max_tokens, "stream": False}
            body.update(json.loads(a.extra) if a.extra else {})
            req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                resp = json.loads(r.read())
            v = judge(c, first_call(resp))
            f.write(json.dumps({"id": c["id"], "expect": c["expect"], **v}) + "\n")
            print(f"  {c['id']:<24} expect {c['expect']:<4} {'called ' + str(v.get('name')) if v['called'] else 'no call':<24} "
                  f"{'right' if v['right'] else 'WRONG'}", flush=True)
    return 0


def load(path) -> dict:
    return {r["id"]: r for r in (json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()) if "id" in r}


def score(path, ref=None) -> dict:
    got = load(path)
    out = {}
    for label in ("call", "none"):
        rows = [r for r in got.values() if r["expect"] == label]
        out[label] = {"right": sum(r["right"] for r in rows), "of": len(rows)}
    out["all"] = {"right": sum(r["right"] for r in got.values()), "of": len(got)}
    if ref is not None:
        base = load(ref)
        shared = sorted(set(got) & set(base))
        same = [i for i in shared if (got[i]["called"], got[i].get("name"), got[i]["right"]) ==
                (base[i]["called"], base[i].get("name"), base[i]["right"])]
        out["agreement"] = {"same": len(same), "of": len(shared), "differ": [i for i in shared if i not in same]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cases", default=str(Path(__file__).with_name("cases.jsonl")))
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--extra", help="JSON merged into every request")
    p.add_argument("--label", default="")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("run")
    p.add_argument("--ref")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    s = score(a.run, a.ref)
    for k in ("call", "none", "all"):
        print(f"{k:<5} {s[k]['right']}/{s[k]['of']}")
    if "agreement" in s:
        g = s["agreement"]
        print(f"agreement with the reference: {g['same']}/{g['of']}" + (f"; differ: {', '.join(g['differ'])}" if g["differ"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
