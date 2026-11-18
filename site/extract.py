#!/usr/bin/env python3
"""Extract every number the site shows from the files in this repository.

The site does not keep its own copy of a measurement.  Each value written by this
script carries the file (and JSON key, table row or log line) it came from.  The
build compares the committed JSON against a fresh extraction and refuses a stale
number, so a number on the page is always a number in an eval, a log, SPECS or a
docs page.

Run `python3 site/extract.py --write` to regenerate site/data, and
`python3 site/extract.py --check` to prove the committed files are current.
"""

from __future__ import annotations

import json
import lzma
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "data"

FORMATS = {
    "text",
    "int",
    "tokens",
    "bytes",
    "mib",
    "gib",
    "gbps",
    "kld",
    "kld_delta",
    "kld_ci",
    "pm",
    "ppl",
    "pct",
    "points",
    "tps",
    "ratio",
    "ms",
    "secs",
    "depth",
    "num2",
}


def format_value(value, fmt="text"):
    """One place where a number becomes text.  Templates never carry a format string."""
    if value is None:
        return "not recorded"
    if fmt == "text":
        return str(value)
    if isinstance(value, str):
        return value
    if fmt == "int":
        return f"{int(round(value)):,}"
    if fmt == "tokens":
        return f"{int(round(value)):,}"
    if fmt == "bytes":
        return f"{int(round(value)):,}"
    if fmt == "mib":
        return f"{value:,.0f} MiB"
    if fmt == "gib":
        return f"{value:.2f} GiB"
    if fmt == "gbps":
        return f"{value:.1f} GB/s"
    if fmt == "kld":
        return f"{value:.5f}"
    if fmt == "kld_delta":
        return f"{value:+.5f}"
    if fmt == "kld_ci":
        lo, hi = value
        return f"[{lo:+.5f}, {hi:+.5f}]"
    if fmt == "pm":
        return f"\u00b1 {value:.5f}"
    if fmt == "ppl":
        return f"{value:.4f}"
    if fmt == "pct":
        return f"{value:.2f}%"
    if fmt == "points":
        return f"{value:+.2f}"
    if fmt == "tps":
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.1f}"
    if fmt == "ratio":
        return f"{value:.1f}\u00d7"
    if fmt == "ms":
        return f"{value:,.0f} ms" if abs(value) >= 100 else f"{value:.1f} ms"
    if fmt == "secs":
        return f"{value:.1f} s"
    if fmt == "depth":
        return f"{int(value):,}"
    if fmt == "num2":
        return f"{value:.2f}"
    raise ValueError(f"unknown format {fmt!r}")


class Dataset:
    def __init__(self, name: str):
        self.name = name
        self.values: dict = {}
        self.sources: dict[str, str] = {}
        self.formats: dict[str, str] = {}

    def add(self, path: str, value, source: str, fmt: str = "text"):
        if fmt not in FORMATS:
            raise ValueError(f"{self.name}: {path}: unknown format {fmt!r}")
        if value is None:
            return
        node = self.values
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{self.name}: {path}: {part} is already a leaf")
        if parts[-1] in node:
            raise ValueError(f"{self.name}: duplicate key {path}")
        node[parts[-1]] = value
        self.sources[path] = source
        self.formats[path] = fmt

    def add_row(self, prefix: str, row: dict, source: str, fmts: dict[str, str]):
        for key, value in row.items():
            if value is None:
                continue
            self.add(f"{prefix}.{key}", value, source, fmts.get(key, "text"))

    def as_json(self) -> dict:
        return {
            "_generated_by": "site/extract.py",
            "_note": "Every value names the file it came from. Do not edit by hand; run python3 site/extract.py --write.",
            "values": self.values,
            "_sources": dict(sorted(self.sources.items())),
            "_formats": dict(sorted(self.formats.items())),
        }


# ---------------------------------------------------------------------------
# small readers


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def read_lines(rel: str) -> list[str]:
    return (ROOT / rel).read_text().splitlines()


def strip_md(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "").replace("\u00a0", " ")
    return text.strip()


def md_tables(rel: str) -> list[dict]:
    lines = read_lines(rel)
    tables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            header = [strip_md(c) for c in line.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [strip_md(c) for c in lines[j].strip().strip("|").split("|")]
                rows.append({"line": j + 1, "cells": cells, "by_header": dict(zip(header, cells))})
                j += 1
            tables.append({"line": i + 1, "header": header, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def table_with(rel: str, needle: str) -> dict:
    for table in md_tables(rel):
        if any(needle in cell for cell in table["header"]):
            return table
    raise LookupError(f"{rel}: no table with a column named {needle!r}")


def row_by_cell(table: dict, column: str, needle: str) -> dict:
    for row in table["rows"]:
        if needle in row["by_header"].get(column, ""):
            return row
    raise LookupError(f"no row with {column}={needle!r}")


def md_number(text: str):
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def code_blocks(rel: str) -> list[dict]:
    lines = read_lines(rel)
    blocks = []
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*```(\w*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        start = i + 1
        lang = m.group(1)
        body = []
        i += 1
        while i < len(lines) and not re.match(r"^\s*```\s*$", lines[i]):
            body.append(lines[i])
            i += 1
        blocks.append({"lang": lang, "line": start, "text": "\n".join(body), "end": i + 1})
        i += 1
    return blocks


def code_block_with(rel: str, needle: str) -> dict:
    for block in code_blocks(rel):
        if needle in block["text"]:
            return block
    raise LookupError(f"{rel}: no code block contains {needle!r}")


def text_at(rel: str, needle: str, pattern: str, group: int = 1) -> tuple[str, str]:
    """Return (text, source) for the first line containing `needle`."""
    for n, line in enumerate(read_lines(rel), 1):
        if needle in line:
            m = re.search(pattern, line)
            if not m:
                break
            return m.group(group), f"{rel} line {n}"
    raise LookupError(f"{rel}: no line with {needle!r} matching {pattern!r}")


def bench(rel: str) -> dict:
    d = read_json(rel)
    summary = {s["test"]: s for s in d.get("summary", [])}
    return {
        "raw": d,
        "commit": d.get("commit"),
        "engine": d.get("engine_version"),
        "model": Path(d.get("model", "")).name,
        "power": d.get("power_limit"),
        "args": d.get("engine_args", ""),
        "gpus": d.get("gpus", []),
        "engine_info": d.get("engine_info", {}),
        "summary": summary,
        "quality": d.get("quality") or {},
    }


def bench_prefix(ds: Dataset, prefix: str, rel: str, b: dict):
    q = b["quality"]
    base = rel
    for key, value, fmt in [
        ("commit", b["commit"], "text"),
        ("engine", b["engine"], "text"),
        ("model", b["model"], "text"),
        ("power_w", md_number(b["power"]) if b["power"] else None, "int"),
        ("args", b["args"], "text"),
    ]:
        ds.add(f"{prefix}.{key}", value, f"{base}#{key}", fmt)
    if b["gpus"]:
        g = b["gpus"][0]
        ds.add(f"{prefix}.gpu", g.get("name"), f"{base}#gpus[0].name")
    for key, fmt in [
        ("chunks", "int"),
        ("rows", "int"),
        ("t", "int"),
        ("ppl", "ppl"),
        ("base_ppl", "ppl"),
        ("kld_mean", "kld"),
        ("kld_median", "kld"),
        ("kld_p99", "kld"),
        ("top1_pct", "pct"),
    ]:
        if key in q and q.get(key) is not None:
            ds.add(f"{prefix}.quality.{key}", q[key], f"{base}#quality.{key}", fmt)
    for test, fmt in [
        ("pp16384", "tps"),
        ("pp32768", "tps"),
        ("pp36864", "tps"),
        ("pp131072", "tps"),
        ("tg conv", "tps"),
        ("tg struct", "tps"),
        ("tg prose", "tps"),
    ]:
        if test in b["summary"]:
            s = b["summary"][test]
            slug = test.replace(" ", "_")
            ds.add(f"{prefix}.speed.{slug}", s["tok_s_mean"], f"{base}#summary[{test}].tok_s_mean", fmt)
            if "accept_pct_mean" in s:
                ds.add(f"{prefix}.accept.{slug}", s["accept_pct_mean"], f"{base}#summary[{test}].accept_pct_mean", "pct")
            if "draft_len_mean" in s:
                ds.add(f"{prefix}.draft_len.{slug}", s["draft_len_mean"], f"{base}#summary[{test}].draft_len_mean", "kld")
    for idx, r in enumerate(b["raw"].get("results", [])):
        if r.get("test") == "tg struct" and "tokens_per_pass" in r:
            ds.add(f"{prefix}.tokens_per_pass", r["tokens_per_pass"], f"{base}#results[{idx}].tokens_per_pass", "num2")
            break


def parse_paired(rel: str) -> dict:
    """paired.md -> {kv: {delta, lo, hi, top1_points, line}}."""
    lines = read_lines(rel)
    out = {}
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 8 or cells[0] in ("KV", "vs k8v4:") or set(cells[0]) <= {"-", ":", " "}:
            continue
        m = re.search(r"([+-]\d*\.\d+)\s*\[\s*([+-]\d*\.\d+)\s*,\s*([+-]\d*\.\d+)\s*\]", cells[6])
        if not m:
            continue
        kv = cells[0]
        if kv.startswith("vs k8v4:"):
            kv = cells[1]
        out[kv] = {
            "delta": float(m.group(1)),
            "lo": float(m.group(2)),
            "hi": float(m.group(3)),
            "points": float(cells[7]) if cells[7] not in ("", "-") else None,
            "line": n,
        }
    # the d8 file has two "vs k8v4:" rows whose first cell is not a table
    for n, line in enumerate(lines, 1):
        m = re.match(r"^vs k8v4:\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            kv = m.group(1).strip()
            dm = re.search(r"([+-]\d*\.\d+)\s*\[\s*([+-]\d*\.\d+)\s*,\s*([+-]\d*\.\d+)\s*\]", m.group(7))
            if dm:
                out[kv + "@vs_k8v4"] = {
                    "delta": float(dm.group(1)),
                    "lo": float(dm.group(2)),
                    "hi": float(dm.group(3)),
                    "points": float(m.group(8)) if m.group(8).strip() not in ("", "-") else None,
                    "line": n,
                    "against": "k8v4",
                }
    return out


def parse_kld_txt(rel: str) -> dict:
    text = (ROOT / rel).read_text()
    out = {}
    for key, pattern in [
        ("mean_kld", r"Mean\s+KLD:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
        ("mean_ppl", r"Mean PPL\(Q\)\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
        ("same_top", r"Same top p:\s*([0-9.]+)\s*±\s*([0-9.]+)"),
    ]:
        m = re.search(pattern, text)
        if not m:
            raise LookupError(f"{rel}: {key} not found")
        out[key] = float(m.group(1))
        out[key + "_err"] = float(m.group(2))
    return out


def parse_smoke(rel: str) -> dict:
    d = read_json(rel)
    r = d["results"]
    out = {"model": d.get("model"), "url": d.get("url")}
    m = re.search(r"warm decode ([\d.]+)-([\d.]+) t/s \(cold ([\d.]+)\)", r["speed"]["detail"])
    if m:
        out["warm_low"], out["warm_high"], out["cold"] = map(float, m.groups())
    for name, key in [("needle", "needle"), ("long", "long")]:
        m = re.search(r"([\d,]+) tok, ([\d.]+) s, prompt ([\d,]+) t/s, decode ([\d.]+) t/s", r[key]["detail"])
        if m:
            out[name] = {
                "tokens": int(m.group(1).replace(",", "")),
                "seconds": float(m.group(2)),
                "prompt": int(m.group(3).replace(",", "")),
                "decode": float(m.group(4)),
            }
    out["details"] = {k: v["detail"] for k, v in r.items()}
    out["ok"] = {k: v["ok"] for k, v in r.items()}
    return out


# ---------------------------------------------------------------------------
# dataset builders


def build_layers() -> Dataset:
    ds = Dataset("layers")
    # The plan's six stops.  The gauge uses the engine's real token counts; the
    # labels are the plan's rounded names.
    layers = [
        ("surface", "Surface", 0, "0"),
        ("sunlit", "Sunlit layer", 8192, "8K"),
        ("twilight", "Twilight", 32768, "32K"),
        ("boundary", "The halocline", 65536, "64K"),
        ("deep", "Deep", 131072, "128K"),
        ("floor", "Floor", 262144, "262K"),
    ]
    for slug, name, depth, label in layers:
        source = "plan: SITE_PLAN.md section 2"
        ds.add(f"{slug}.name", name, source)
        ds.add(f"{slug}.depth", depth, source, "depth")
        ds.add(f"{slug}.label", label, source)
    ds.add("top_depth", 262144, "plan: SITE_PLAN.md section 2", "depth")
    return ds


def build_machine() -> Dataset:
    ds = Dataset("machine")
    spec = "SPECS.md"
    t = md_tables(spec)
    hw = t[0]
    sw = t[1]
    models = t[2]

    def row(table, first):
        return row_by_cell(table, table["header"][0], first)

    gpu = row(hw, "GPU")
    gpu_text = gpu["by_header"][hw["header"][1]]
    m = re.search(r"(NVIDIA GeForce RTX \d+), ([\d,]+) MiB, compute capability ([\d.]+)", gpu_text)
    ds.add("gpu.name", m.group(1), f"{spec} line {gpu['line']}")
    ds.add("gpu.vram_mib", float(m.group(2).replace(",", "")), f"{spec} line {gpu['line']}", "mib")
    ds.add("gpu.compute_cap", m.group(3), f"{spec} line {gpu['line']}")

    pw = row(hw, "GPU power limit")
    ds.add("gpu.power_w", md_number(pw["by_header"][hw["header"][1]]), f"{spec} line {pw['line']}", "int")

    link = row(hw, "GPU link")
    ds.add("gpu.pcie", link["by_header"][hw["header"][1]], f"{spec} line {link['line']}")

    clocks = row(hw, "GPU clocks (max)")
    cm = re.search(r"SM ([\d,]+) MHz, memory ([\d,]+) MHz", clocks["by_header"][hw["header"][1]])
    ds.add("gpu.sm_mhz", float(cm.group(1).replace(",", "")), f"{spec} line {clocks['line']}", "int")
    ds.add("gpu.mem_mhz", float(cm.group(2).replace(",", "")), f"{spec} line {clocks['line']}", "int")

    cpu = row(hw, "CPU")
    ds.add("host.cpu", row(hw, "CPU")["by_header"][hw["header"][1]].split(",")[0], f"{spec} line {cpu['line']}")
    tm = re.search(r"(\d+) threads", cpu["by_header"][hw["header"][1]])
    ds.add("host.threads", float(tm.group(1)), f"{spec} line {cpu['line']}", "int")

    ram = row(hw, "RAM")
    rm = re.search(r"(\d+) GB", ram["by_header"][hw["header"][1]])
    ds.add("host.ram_gb", float(rm.group(1)), f"{spec} line {ram['line']}", "int")

    os_row = row(sw, "OS")
    ds.add("software.os", os_row["by_header"][sw["header"][1]].split(",")[0], f"{spec} line {os_row['line']}")
    km = re.search(r"kernel ([\d.]+)", os_row["by_header"][sw["header"][1]])
    ds.add("software.kernel", km.group(1), f"{spec} line {os_row['line']}")

    for key, first in [("driver", "NVIDIA driver"), ("cuda", "CUDA toolkit"), ("gcc", "Host compiler"), ("cmake", "CMake")]:
        r = row(sw, first)
        ds.add(f"software.{key}", r["by_header"][sw["header"][1]], f"{spec} line {r['line']}")

    sparse = row(models, "Qwen3.8-Flash-Next")
    dense = row(models, "Qwen3.8-27B")
    ds.add("models.sparse.name", "Qwen3.8-Flash-Next", f"{spec} line {sparse['line']}")
    ds.add("models.dense.name", "Qwen3.8-27B", f"{spec} line {dense['line']}")
    ds.add("models.sparse.arch", sparse["by_header"][models["header"][0]], f"{spec} line {sparse['line']}")
    ds.add("models.dense.arch", dense["by_header"][models["header"][0]], f"{spec} line {dense['line']}")
    ds.add("models.sparse.role", sparse["by_header"][models["header"][2]], f"{spec} line {sparse['line']}")

    # README facts
    readme = "README.md"
    lines = read_lines(readme)
    for n, line in enumerate(lines, 1):
        if "status: pre-alpha" in line:
            ds.add("project.status", "pre-alpha", f"{readme} line {n}")
            ds.add("project.licence", "MIT", f"{readme} line {n}")
        if "93,672,359,738 bytes of weights" in line:
            ds.add("models.sparse.weights_bytes", 93672359738, f"{readme} line {n}", "bytes")
        if "stands in for Qwen4-27B" in line:
            ds.add("models.dense.target", "Qwen4-27B", f"{readme} line {n}")
        if "125B parameters with 6B active" in line:
            ds.add("models.sparse.params_b", 125, f"{readme} line {n}", "int")
            ds.add("models.sparse.active_b", 6, f"{readme} line {n}", "int")
        if "122,880 served" in line:
            ds.add("models.dense.served_context", 122880, f"{readme} line {n}", "tokens")

    # commands: the README's own blocks, not retyped
    for key, needle in [
        ("build", "cmake -S . -B build -G Ninja"),
        ("test", "ctest --test-dir build -L cpu"),
        ("convert", "tools/convert/convert.py --gguf Qwen3.8-27B"),
        ("serve_dense", "bin/halocline serve Qwen3.8-27B"),
        ("quality", "bin/halocline-bench --model Qwen3.8-27B"),
    ]:
        block = code_block_with(readme, needle)
        ds.add(f"commands.{key}.text", block["text"], f"{readme} line {block['line']}")
        ds.add(f"commands.{key}.line", block["line"], f"{readme} line {block['line']}", "int")

    # gate counts, from the page that explains them
    testing = "docs/testing.md"
    tt = table_with(testing, "Gate")
    for first, key in [("Build", "build"), ("SASS per object and per function", "sass"), ("CPU suite", "cpu"), ("GPU suite", "gpu"), ("Python server suite", "python")]:
        r = row_by_cell(tt, "Gate", first)
        ds.add(f"gates.{key}.result", r["by_header"]["Current result"], f"{testing} line {r['line']}")

    # format names, so the prose can name them without typing a number
    ds.add("formats.bf16", "BF16", f"{readme} line 33")
    ds.add("formats.q8_0", "Q8_0", "evals/2026-11-17/halocline-dense-load/README.md line 31")
    ds.add("labels.p99", "p99", "docs/precision/quality.md line 19")
    ds.add("credits.strata_author", "Niko1221", f"{readme} line 329")
    ds.add("credits.basalt_author", "Jes\u00fas Gulfo", f"{readme} line 331")
    ds.add("credits.infernix_author", "Wallawalla47", f"{readme} line 333")
    return ds


def build_sparse() -> Dataset:
    ds = Dataset("sparse")
    readme = "README.md"
    lines = read_lines(readme)

    # architecture facts, from the SPECS paragraph (it wraps across three lines)
    spec = read_lines("SPECS.md")
    for n, line in enumerate(spec, 1):
        if line.startswith("Model facts for the sparse path"):
            para = " ".join(spec[n - 1 : n + 2])
            m = re.search(
                r"(\d+) layers, (\d+) cycles of three Gated DeltaNet layers and one Qwen Sparse Attention layer, "
                r"hidden size ([\d,]+), (\d+) experts with (\d+) routed and (\d+) shared active per token, "
                r"expert intermediate dimension ([\d,]+), n-gram vocabulary ([\d,]+) at layer (\d+), ([\d,]+) native context",
                para,
            )
            if not m:
                raise LookupError("SPECS architecture paragraph did not match")
            src = f"SPECS.md line {n}"
            ds.add("facts.layers", float(m.group(1)), src, "int")
            ds.add("facts.cycles", float(m.group(2)), src, "int")
            ds.add("facts.hidden", float(m.group(3).replace(",", "")), src, "int")
            ds.add("facts.experts", float(m.group(4)), src, "int")
            ds.add("facts.active_routed", float(m.group(5)), src, "int")
            ds.add("facts.shared", float(m.group(6)), src, "int")
            ds.add("facts.expert_intermediate", float(m.group(7).replace(",", "")), src, "int")
            ds.add("facts.ngram_vocab", float(m.group(8).replace(",", "")), src, "int")
            ds.add("facts.ngram_layer", float(m.group(9)), src, "int")
            ds.add("facts.native_context", float(m.group(10).replace(",", "")), src, "tokens")

    for n, line in enumerate(lines, 1):
        if "51B" in line and "n-gram embedding table" in line:
            ds.add("facts.ngram_params_b", 51, f"{readme} line {n}", "int")
            ds.add("facts.mtp_params_b", 4, f"{readme} line {n}", "int")
        if "195 `hc_*` tensors" in line and "facts.hc_tensors" not in ds.sources:
            ds.add("facts.hc_tensors", 195, f"{readme} line {n}", "int")

    # pack facts, from the quality eval README
    kvreadme = "evals/2026-11-17/halocline-kv-kld/README.md"
    for n, line in enumerate(read_lines(kvreadme), 1):
        if "90.08 GiB, 15 sections, 343 s" in line:
            m = re.search(r"([\d.]+) GiB, (\d+) sections, (\d+) s", line)
            if not m:
                raise LookupError("the pack line did not match")
            ds.add("pack.size_gib", float(m.group(1)), f"{kvreadme} line {n}", "gib")
            ds.add("pack.sections", float(m.group(2)), f"{kvreadme} line {n}", "int")
            ds.add("pack.seconds", float(m.group(3)), f"{kvreadme} line {n}", "int")

    # placement, from the published build's own log
    err = "evals/2026-11-17/halocline-greedy-identity/new.err"
    text = (ROOT / err).read_text().splitlines()
    for n, line in enumerate(text, 1):
        m = re.search(r"resident RAM mode: ([\d.]+) GiB of experts in RAM \(page-locked\), (\d+) in the GPU cache", line)
        if m:
            ds.add("placement.ram_gib", float(m.group(1)), f"{err} line {n}", "gib")
            ds.add("placement.expert_slots", float(m.group(2)), f"{err} line {n}", "int")
        m = re.search(r"experts live in VRAM ([\d.]+) GiB \((\d+)%\).*RAM ([\d.]+) GiB \((\d+)%\).*file \(SSD\) ([\d.]+) GiB", line)
        if m:
            ds.add("placement.vram_gib", float(m.group(1)), f"{err} line {n}", "gib")
            ds.add("placement.vram_pct", float(m.group(2)), f"{err} line {n}", "pct")
            ds.add("placement.ram_tier_gib", float(m.group(3)), f"{err} line {n}", "gib")
            ds.add("placement.ram_pct", float(m.group(4)), f"{err} line {n}", "pct")
            ds.add("placement.ssd_gib", float(m.group(5)), f"{err} line {n}", "gib")
        m = re.search(r"PCIe probe: ([\d.]+) GB/s host->device", line)
        if m:
            ds.add("placement.pcie_gbps", float(m.group(1)), f"{err} line {n}", "gbps")
        m = re.search(r"(\d+) MiB of VRAM free with everything loaded", line)
        if m:
            ds.add("placement.vram_free_mib", float(m.group(1)), f"{err} line {n}", "mib")

    # KV arms
    kv_dir = "evals/2026-11-17/halocline-kv-kld"
    paired = parse_paired(f"{kv_dir}/paired.md")
    for arm, label in [("int8", "8-bit"), ("fp16", "16-bit"), ("k8v4", "K 8-bit, V 4-bit"), ("q4_0", "4-bit")]:
        rel = f"{kv_dir}/{arm}/bench.json"
        b = bench(rel)
        prefix = f"kv.arms.{arm}"
        ds.add(f"{prefix}.flag", re.search(r"--kv\s+(\S+)", b["args"]).group(1), f"{rel}#engine_args")
        ds.add(f"{prefix}.label", label, f"{rel}#engine_args")
        bench_prefix(ds, prefix, rel, b)
        if arm in paired:
            p = paired[arm]
            src = f"{kv_dir}/paired.md line {p['line']}"
            ds.add(f"{prefix}.delta", p["delta"], src, "kld_delta")
            ds.add(f"{prefix}.ci", [p["lo"], p["hi"]], src, "kld_ci")
            if p["points"] is not None:
                ds.add(f"{prefix}.points", p["points"], src, "points")
    base_bench = bench(f"{kv_dir}/int8/bench.json")
    ds.add("quality.mean_kld", base_bench["quality"]["kld_mean"], f"{kv_dir}/int8/bench.json#quality.kld_mean", "kld")
    ds.add("quality.top1", base_bench["quality"]["top1_pct"], f"{kv_dir}/int8/bench.json#quality.top1_pct", "pct")
    ds.add("quality.chunks", base_bench["quality"]["chunks"], f"{kv_dir}/int8/bench.json#quality.chunks", "int")
    ds.add("quality.rows", base_bench["quality"]["rows"], f"{kv_dir}/int8/bench.json#quality.rows", "int")
    text, src = text_at(f"{kv_dir}/README.md", "Pack: `/data/models/halocline/packs/QFN-UD-IQ4_XS", r"QFN-UD-IQ4_XS-(v[\d.]+)")
    ds.add("quant.label", f"UD-IQ4_XS {text}", src)
    for depth, needle, which in [("32", "| prefill 32K t/s", 0), ("131", "| prefill 131K t/s", -1)]:
        for n, line in enumerate(read_lines(f"{kv_dir}/README.md"), 1):
            if needle in line:
                matches = re.findall(r"prefill (\d+K) t/s", line)
                if not matches:
                    raise LookupError(f"{kv_dir}/README.md: no prefill label on line {n}")
                ds.add(f"depth.{depth}.label", matches[which], f"{kv_dir}/README.md line {n}")
                break
        else:
            raise LookupError(f"{kv_dir}/README.md: no line with {needle!r}")

    # draft window arms
    for spec_n in (5, 8, 12):
        rel = f"{kv_dir}/spec{spec_n}/bench.json"
        b = bench(rel)
        prefix = f"draft.arms.{spec_n}"
        ds.add(f"{prefix}.label", str(spec_n), f"{rel}#engine_args")
        bench_prefix(ds, prefix, rel, b)

    # serve smoke, drafter on through llama-swap
    rel = "evals/2026-11-17/halocline-llama-swap/smoke.json"
    sm = parse_smoke(rel)
    prefix = "serve"
    ds.add(f"{prefix}.warm_low", sm["warm_low"], f"{rel}#results.speed.detail", "tps")
    ds.add(f"{prefix}.warm_high", sm["warm_high"], f"{rel}#results.speed.detail", "tps")
    ds.add(f"{prefix}.cold", sm["cold"], f"{rel}#results.speed.detail", "tps")
    ds.add(f"{prefix}.needle.decode", sm["needle"]["decode"], f"{rel}#results.needle.detail", "tps")
    ds.add(f"{prefix}.needle.prompt", sm["needle"]["prompt"], f"{rel}#results.needle.detail", "tps")
    ds.add(f"{prefix}.long.decode", sm["long"]["decode"], f"{rel}#results.long.detail", "tps")
    ds.add(f"{prefix}.long.prompt", sm["long"]["prompt"], f"{rel}#results.long.detail", "tps")

    # the same pack served without the drafter
    rel = "evals/2026-11-17/halocline-sparse-serve/smoke-1.json"
    sm = parse_smoke(rel)
    prefix = "no_drafter"
    for key, fmt in [("warm_low", "tps"), ("warm_high", "tps"), ("cold", "tps")]:
        ds.add(f"{prefix}.{key}", sm[key], f"{rel}#results.speed.detail", fmt)
    for depth, key in [("needle", "needle"), ("long", "long")]:
        ds.add(f"{prefix}.{depth}.decode", sm[key]["decode"], f"{rel}#results.{key}.detail", "tps")
        ds.add(f"{prefix}.{depth}.prompt", sm[key]["prompt"], f"{rel}#results.{key}.detail", "tps")

    # the in-tree bench at three depths, no drafter (different protocol)
    rel = "evals/2026-11-17/halocline-sparse-serve/basalt-bench-1.txt"
    bt = read_lines(rel)
    for n, line in enumerate(bt, 1):
        m = re.match(r"\s+run 1 pp(\d+) (\d+) tokens, ([\d,]+) t/s engine", line)
        if m:
            depth = int(m.group(1))
            ds.add(f"bench.pp{depth}", float(m.group(3).replace(",", "")), f"{rel} line {n}", "tps")
        m = re.match(r"\s+run 1 (conv|struct|prose)\s+([\d.]+) tok/s", line)
        if m:
            ds.add(f"bench.{m.group(1)}", float(m.group(2)), f"{rel} line {n}", "tps")

    return ds


def build_dense() -> Dataset:
    ds = Dataset("dense")
    roadmap = "docs/roadmap.md"
    for n, line in enumerate(read_lines(roadmap), 1):
        if "64 layers, hidden size" in line:
            m = re.search(r"(\d+) layers, hidden size ([\d,]+)", line)
            if not m:
                raise LookupError("roadmap architecture line did not match")
            src = f"{roadmap} line {n}"
            ds.add("facts.layers", float(m.group(1)), src, "int")
            ds.add("facts.hidden", float(m.group(2).replace(",", "")), src, "int")
        if "SwiGLU FFN in each" in line:
            m = re.search(r"([\d,]+) vocabulary, ([\d,]+) native context", line)
            if not m:
                raise LookupError("roadmap vocabulary line did not match")
            src = f"{roadmap} line {n}"
            ds.add("facts.vocab", float(m.group(1).replace(",", "")), src, "int")
            ds.add("facts.native_context", float(m.group(2).replace(",", "")), src, "tokens")

    # load facts, from the engine's own stderr
    for rel, pack in [
        ("evals/2026-11-17/halocline-dense-load/iq4xs.err", "iq4xs"),
        ("evals/2026-11-17/halocline-dense-load/q4kxl.err", "q4kxl"),
    ]:
        label = "UD-IQ4_XS" if pack == "iq4xs" else "UD-Q4_K_XL"
        ds.add(f"packs.{pack}.label", label, rel)
        for n, line in enumerate(read_lines(rel), 1):
            m = re.search(r"([\d.]+) GiB on the GPU, ([\d.]+) GiB of embedding in page-locked RAM", line)
            if m:
                ds.add(f"packs.{pack}.gpu_gib", float(m.group(1)), f"{rel} line {n}", "gib")
                ds.add(f"packs.{pack}.embedding_gib", float(m.group(2)), f"{rel} line {n}", "gib")
            m = re.search(r"batch(?:ed)? prefill (\d+) tokens in (\d+) ms", line)
            if m:
                ds.add(f"packs.{pack}.prefill_tokens", float(m.group(1)), f"{rel} line {n}", "tokens")
                ds.add(f"packs.{pack}.prefill_ms", float(m.group(2)), f"{rel} line {n}", "ms")
            m = re.search(r"prompt \d+ ms, (\d+) tokens in (\d+) ms", line)
            if m:
                ds.add(f"packs.{pack}.decode_tokens", float(m.group(1)), f"{rel} line {n}", "tokens")
                ds.add(f"packs.{pack}.decode_ms", float(m.group(2)), f"{rel} line {n}", "ms")
            m = re.search(r"(\d+) tensors", line)
            if m:
                ds.add(f"packs.{pack}.tensors", float(m.group(1)), f"{rel} line {n}", "int")

    # quality arms
    qdir = "evals/2026-11-17/halocline-dense-quality"
    # the depth grid of the dense protocol
    d4 = f"{qdir}/d4.sh"
    for n, line in enumerate(read_lines(d4), 1):
        m = re.search(r"-p\s+(\d+),(\d+)", line)
        if m:
            ds.add("depth16.tokens", float(m.group(1)), f"{d4} line {n}", "tokens")
            ds.add("depth36.tokens", float(m.group(2)), f"{d4} line {n}", "tokens")

    arms = {
        "halo_q4kxl_int8": (f"{qdir}/q4kxl-int8-mmvq/bench.json", "Halocline", "UD-Q4_K_XL", "mmvq"),
        "halo_iq4xs_int8": (f"{qdir}/iq4xs-int8/bench.json", "Halocline", "UD-IQ4_XS", "MMQ"),
        "halo_iq4xs_mmvq": (f"{qdir}/iq4xs-int8-mmvq/bench.json", "Halocline", "UD-IQ4_XS", "mmvq"),
    }
    for key, (rel, engine, quant, mode) in arms.items():
        b = bench(rel)
        prefix = f"quality.arms.{key}"
        ds.add(f"{prefix}.label", f"{engine}, {quant}, {mode}", f"{rel}#engine_args")
        ds.add(f"{prefix}.quant", quant, f"{rel}#engine_args")
        ds.add(f"{prefix}.mode", mode, f"{rel}#engine_args")
        bench_prefix(ds, prefix, rel, b)
    for key, rel, quant, kv, fmt in [
        ("llama_q4kxl_f16", f"{qdir}/llama-q4kxl-f16/kld.txt", "UD-Q4_K_XL", "f16", "llama"),
        ("llama_iq4xs_f16", f"{qdir}/llama-iq4xs-f16/kld.txt", "UD-IQ4_XS", "f16", "llama"),
        ("llama_iq4xs_q8", f"{qdir}/llama-iq4xs-q8/kld.txt", "UD-IQ4_XS", "q8_0", "llama"),
    ]:
        k = parse_kld_txt(rel)
        prefix = f"quality.arms.{key}"
        ds.add(f"{prefix}.engine", "llama.cpp", rel)
        ds.add(f"{prefix}.label", f"llama.cpp, {quant}, {kv}", rel)
        ds.add(f"{prefix}.quant", quant, rel)
        ds.add(f"{prefix}.kv", kv, rel)
        ds.add(f"{prefix}.mean_kld", k["mean_kld"], f"{rel}#Mean KLD", "kld")
        ds.add(f"{prefix}.err", k["mean_kld_err"], f"{rel}#Mean KLD", "pm")
        ds.add(f"{prefix}.ci", [-k["mean_kld_err"], k["mean_kld_err"]], f"{rel}#Mean KLD", "kld_ci")
        ds.add(f"{prefix}.mean_ppl", k["mean_ppl"], f"{rel}#Mean PPL(Q)", "ppl")
        ds.add(f"{prefix}.top1_pct", k["same_top"], f"{rel}#Same top p", "pct")

    ds.add(
        "quality.ratio",
        ds.values["quality"]["arms"]["halo_iq4xs_int8"]["quality"]["kld_mean"]
        / ds.values["quality"]["arms"]["halo_q4kxl_int8"]["quality"]["kld_mean"],
        "computed: the two D4 arms in this run",
        "ratio",
    )

    text, src = text_at(f"{qdir}/README.md", "llama.cpp `2cbe0ec`", r"llama\.cpp `([0-9a-f]+)`")
    ds.add("quality.llama_commit", text, src)

    # mmvq cost, from the eval README's paired arm
    dense_readme = f"{qdir}/README.md"
    for n, line in enumerate(read_lines(dense_readme), 1):
        if "mmvq costs +0.00015 mean KLD" in line:
            ds.add("quality.mmvq_delta", 0.00015, f"{dense_readme} line {n}", "kld_delta")
            ds.add("quality.mmvq_ci", [0.00005, 0.00025], f"{dense_readme} line {n}", "kld_ci")
        if "caps mmvq at 5 rows" in line:
            ds.add("quality.mmvq_rows", 5, f"{dense_readme} line {n}", "int")

    # KV arms, 36K gate
    kdir = "evals/2026-11-17/halocline-d8-kv"
    paired = parse_paired(f"{kdir}/paired.md")
    for arm, flag, label in [
        ("int8", "int8", "8-bit"),
        ("k8v4", "k8v4", "K 8-bit, V 4-bit"),
        ("k8v4-mtpint8", "k8v4", "K 8-bit, V 4-bit, drafter 8-bit"),
        ("q4", "q4_0", "4-bit K and V"),
    ]:
        rel = f"{kdir}/{arm}/bench.json"
        b = bench(rel)
        prefix = f"kv.arms.{arm}"
        ds.add(f"{prefix}.flag", flag, f"{rel}#engine_args")
        ds.add(f"{prefix}.label", label, f"{rel}#engine_args")
        ds.add(f"{prefix}.short", {"int8": "int8", "k8v4": "K8V4", "k8v4-mtpint8": "K8V4", "q4": "q4_0"}[arm], f"{kdir}/README.md line 1")
        bench_prefix(ds, prefix, rel, b)
        if arm in paired:
            p = paired[arm]
            src = f"{kdir}/paired.md line {p['line']}"
            ds.add(f"{prefix}.delta", p["delta"], src, "kld_delta")
            ds.add(f"{prefix}.ci", [p["lo"], p["hi"]], src, "kld_ci")
            if p.get("points") is not None:
                ds.add(f"{prefix}.points", p["points"], src, "points")
        if arm + "@vs_k8v4" in paired:
            p = paired[arm + "@vs_k8v4"]
            src = f"{kdir}/paired.md line {p['line']}"
            ds.add(f"{prefix}.delta_vs_k8v4", p["delta"], src, "kld_delta")
            ds.add(f"{prefix}.ci_vs_k8v4", [p["lo"], p["hi"]], src, "kld_ci")

    # prefill ABBA
    vals = {}
    for arm in ("int8", "k8v4"):
        for half in ("a", "b"):
            rel = f"{kdir}/pf-{arm}-{half}/bench.json"
            b = bench(rel)
            vals[(arm, half)] = (b["summary"]["pp16384"]["tok_s_mean"], b["summary"]["pp36864"]["tok_s_mean"], rel)
    for arm in ("int8", "k8v4"):
        a, b = vals[(arm, "a")], vals[(arm, "b")]
        ds.add(f"prefill.{arm}.pp16k_a", a[0], f"{a[2]}#summary[pp16384].tok_s_mean", "tps")
        ds.add(f"prefill.{arm}.pp16k_b", b[0], f"{b[2]}#summary[pp16384].tok_s_mean", "tps")
        ds.add(f"prefill.{arm}.pp36k_a", a[1], f"{a[2]}#summary[pp36864].tok_s_mean", "tps")
        ds.add(f"prefill.{arm}.pp36k_b", b[1], f"{b[2]}#summary[pp36864].tok_s_mean", "tps")
        ds.add(f"prefill.{arm}.pp16k_mean", (a[0] + b[0]) / 2, "computed: mean of the two ABBA arms", "tps")
        ds.add(f"prefill.{arm}.pp36k_mean", (a[1] + b[1]) / 2, "computed: mean of the two ABBA arms", "tps")
    d16 = (vals[("k8v4", "a")][0] + vals[("k8v4", "b")][0]) / 2 / ((vals[("int8", "a")][0] + vals[("int8", "b")][0]) / 2) - 1
    d36 = (vals[("k8v4", "a")][1] + vals[("k8v4", "b")][1]) / 2 / ((vals[("int8", "a")][1] + vals[("int8", "b")][1]) / 2) - 1
    ds.add("prefill.delta_16k_pct", d16 * 100, "computed: ABBA means", "pct")
    ds.add("prefill.delta_36k_pct", d36 * 100, "computed: ABBA means", "pct")

    # the served context and the planner's context ceilings, from the docs table
    prec = "docs/precision/qwen38-27b.md"
    pt = table_with(prec, "Format")
    for fmt_name, slug in [("fp16", "fp16"), ("int8", "int8"), ("k8v4", "k8v4"), ("q4_0 (branch", "q4")]:
        r = row_by_cell(pt, "Format", fmt_name)
        cells = r["by_header"]
        src = f"{prec} line {r['line']}"
        ds.add(f"context.{slug}.cell_bytes", float(cells["Bytes a cell (one attention layer)"].replace(",", "")), src, "bytes")
        ds.add(f"context.{slug}.label", cells["Format"], src)
        ctx = cells["Max context on UD-Q4_K_XL, MTP and vision"]
        first = ctx.split(";")[0].replace("*", "").strip()
        ds.add(f"context.{slug}.max_first", float(first.replace(",", "")), src, "tokens")
        ds.add(f"context.{slug}.text", ctx, src)
        mtp = re.search(r"([\d,]+) with --mtp-kv int8", ctx)
        if mtp:
            ds.add(f"context.{slug}.max_mtp", float(mtp.group(1).replace(",", "")), src, "tokens")
        ring = re.search(r"([\d,]+) adding the (?:32K )?(?:drafter )?ring", ctx)
        if ring:
            ds.add(f"context.{slug}.max_ring", float(ring.group(1).replace(",", "")), src, "tokens")
    # the 262K fit probe
    for rel, pack in [
        (f"{qdir}/fit/UD-IQ4_XS-int8.err", "iq4xs"),
        (f"{qdir}/fit/UD-Q4_K_XL-int8.err", "q4kxl"),
    ]:
        for n, line in enumerate(read_lines(rel), 1):
            m = re.search(r"1 session\(s\) of (\d+) cells need (\d+) MiB, (\d+) MiB are free", line)
            if m:
                ds.add(f"fit.{pack}.need_mib", float(m.group(2)), f"{rel} line {n}", "mib")
                ds.add(f"fit.{pack}.free_mib", float(m.group(3)), f"{rel} line {n}", "mib")
                ds.add(f"fit.{pack}.cells", float(m.group(1)), f"{rel} line {n}", "tokens")
    ds.add("fit.q4kxl.free_vision_mib", 7597, f"{prec} line 75", "mib")
    ds.add("fit.iq4xs.free_vision_mib", 10589, f"{prec} line 75", "mib")
    ds.add("fit.fp16_need_mib", 19766, f"{prec} line 77", "mib")
    for depth, text in [("16", "16K"), ("36", "36K")]:
        ds.add(f"prefill.depth{depth}.label", text, f"{kdir}/README.md line 51")

    # the dense speed floor, NInfer
    rel = "evals/2026-11-16/halocline-p0-floors/ninfer-1.json"
    sm = parse_smoke(rel)
    ds.add("floor.warm_low", sm["warm_low"], f"{rel}#results.speed.detail", "tps")
    ds.add("floor.warm_high", sm["warm_high"], f"{rel}#results.speed.detail", "tps")
    ds.add("floor.needle.decode", sm["needle"]["decode"], f"{rel}#results.needle.detail", "tps")
    ds.add("floor.needle.prompt", sm["needle"]["prompt"], f"{rel}#results.needle.detail", "tps")
    ds.add("floor.long.decode", sm["long"]["decode"], f"{rel}#results.long.detail", "tps")
    ds.add("floor.long.prompt", sm["long"]["prompt"], f"{rel}#results.long.detail", "tps")

    # the first load
    rel = "evals/2026-11-17/halocline-dense-load/README.md"
    for n, line in enumerate(read_lines(rel), 1):
        if "866/866 LOSSLESS" in line:
            ds.add("load.verify", 866, f"{rel} line {n}", "int")
            ds.add("load.verify_total", 866, f"{rel} line {n}", "int")

    return ds


def build_compare() -> Dataset:
    ds = Dataset("compare")
    # the depth labels the tables use, from the eval READMEs that name them
    ds.add("sparse.depth36.label", "36K", "evals/2026-11-17/halocline-llama-swap/README.md line 15")
    ds.add("sparse.depth137.label", "137K", "evals/2026-11-17/halocline-llama-swap/README.md line 16")
    ds.add("dense.depth16.label", "16K", "evals/2026-11-17/halocline-d8-kv/README.md line 51")
    ds.add("dense.depth36.label", "36K", "evals/2026-11-17/halocline-d8-kv/README.md line 51")
    ds.add("dense.depth137.label", "137K", "evals/2026-11-16/halocline-p0-floors/README.md line 31")
    text, src = text_at("evals/2026-11-17/halocline-llama-swap/README.md", "--kv int8", r"--kv\s+([a-z0-9_]+)")
    ds.add("sparse.halocline.kv", text, src)
    text, src = text_at("evals/2026-11-16/halocline-p0-floors/README.md", "q4_0 KV", r"(\S+) KV")
    ds.add("sparse.strata.kv", text, src)
    text, src = text_at("evals/2026-11-16/halocline-p0-floors/README.md", "Strata 0.2.10", r"Strata ([\d.]+)")
    ds.add("sparse.strata.version", text, src)
    text, src = text_at("evals/2026-11-16/halocline-p0-floors/README.md", "DFlash2", r"(DFlash2)")
    ds.add("sparse.strata.drafter", text, src)
    # sparse: the same query through llama-swap, and the floors
    for name, rel in [
        ("halocline", "evals/2026-11-17/halocline-llama-swap/smoke.json"),
        ("no_drafter", "evals/2026-11-17/halocline-sparse-serve/smoke-1.json"),
    ]:
        sm = parse_smoke(rel)
        ds.add(f"sparse.{name}.warm_low", sm["warm_low"], f"{rel}#results.speed.detail", "tps")
        ds.add(f"sparse.{name}.warm_high", sm["warm_high"], f"{rel}#results.speed.detail", "tps")
        for depth, key in [("36k", "needle"), ("137k", "long")]:
            ds.add(f"sparse.{name}.{depth}.decode", sm[key]["decode"], f"{rel}#results.{key}.detail", "tps")
            ds.add(f"sparse.{name}.{depth}.prompt", sm[key]["prompt"], f"{rel}#results.{key}.detail", "tps")
            ds.add(f"sparse.{name}.{depth}.tokens", sm[key]["tokens"], f"{rel}#results.{key}.detail", "tokens")
    rel = "evals/2026-11-16/halocline-p0-floors/strata/2r-df2.json"
    sm = parse_smoke(rel)
    ds.add("sparse.strata.warm_low", sm["warm_low"], f"{rel}#results.speed.detail", "tps")
    ds.add("sparse.strata.warm_high", sm["warm_high"], f"{rel}#results.speed.detail", "tps")
    for depth, key in [("36k", "needle"), ("137k", "long")]:
        ds.add(f"sparse.strata.{depth}.decode", sm[key]["decode"], f"{rel}#results.{key}.detail", "tps")
        ds.add(f"sparse.strata.{depth}.prompt", sm[key]["prompt"], f"{rel}#results.{key}.detail", "tps")
        ds.add(f"sparse.strata.{depth}.tokens", sm[key]["tokens"], f"{rel}#results.{key}.detail", "tokens")

    # dense: Halocline's own prefill and NInfer's floor
    rel = "evals/2026-11-17/halocline-d8-kv/int8/bench.json"
    b = bench(rel)
    ds.add("dense.halocline.pp16k", b["summary"]["pp16384"]["tok_s_mean"], f"{rel}#summary[pp16384].tok_s_mean", "tps")
    ds.add("dense.halocline.pp36k", b["summary"]["pp36864"]["tok_s_mean"], f"{rel}#summary[pp36864].tok_s_mean", "tps")
    ds.add("dense.halocline.conv", b["summary"]["tg conv"]["tok_s_mean"], f"{rel}#summary[tg conv].tok_s_mean", "tps")
    ds.add("dense.halocline.struct", b["summary"]["tg struct"]["tok_s_mean"], f"{rel}#summary[tg struct].tok_s_mean", "tps")
    ds.add("dense.halocline.prose", b["summary"]["tg prose"]["tok_s_mean"], f"{rel}#summary[tg prose].tok_s_mean", "tps")
    rel = "evals/2026-11-16/halocline-p0-floors/ninfer-1.json"
    sm = parse_smoke(rel)
    ds.add("dense.ninfer.warm_low", sm["warm_low"], f"{rel}#results.speed.detail", "tps")
    ds.add("dense.ninfer.warm_high", sm["warm_high"], f"{rel}#results.speed.detail", "tps")
    ds.add("dense.ninfer.pp36k", sm["needle"]["prompt"], f"{rel}#results.needle.detail", "tps")
    ds.add("dense.ninfer.decode36k", sm["needle"]["decode"], f"{rel}#results.needle.detail", "tps")
    ds.add("dense.ninfer.tokens36k", sm["needle"]["tokens"], f"{rel}#results.needle.detail", "tokens")
    ds.add("dense.ninfer.pp137k", sm["long"]["prompt"], f"{rel}#results.long.detail", "tps")
    ds.add("dense.ninfer.decode137k", sm["long"]["decode"], f"{rel}#results.long.detail", "tps")
    ds.add("dense.ninfer.tokens137k", sm["long"]["tokens"], f"{rel}#results.long.detail", "tokens")
    return ds


def build_regress() -> Dataset:
    ds = Dataset("regress")
    root = "evals/2026-11-17/halocline-sparse-regress"
    # the saved summary says FAIL for gate 2; the README records the quiet-host rerun
    summary = f"{root}/summary.md"
    for n, line in enumerate(read_lines(summary), 1):
        if line.startswith("| 2 | KLD + speed"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ds.add("gate2.saved_verdict", cells[2], f"{summary} line {n}")
            m = re.search(r"worst speed delta ([\d.]+)% \(limit (\d+)%\)", cells[3])
            if m:
                ds.add("gate2.first_delta_pct", float(m.group(1)), f"{summary} line {n}", "pct")
                ds.add("gate2.limit_pct", float(m.group(2)), f"{summary} line {n}", "pct")
        if line.startswith("| 1 | greedy identity"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ds.add("gate1.verdict", cells[2], f"{summary} line {n}")
        if line.startswith("| 3 | sampling"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ds.add("gate3.verdict", cells[2], f"{summary} line {n}")
        if line.startswith("| 4 | tool-call"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ds.add("gate4.verdict", cells[2], f"{summary} line {n}")

    rel = f"{root}/1-greedy/greedy.txt"
    text = (ROOT / rel).read_text()
    m = re.search(r"ref: (\d+) tokens", text)
    ds.add("greedy.ref_tokens", float(m.group(1)), rel, "tokens")
    m = re.search(r"new: (\d+) tokens", text)
    ds.add("greedy.new_tokens", float(m.group(1)), rel, "tokens")
    ds.add("greedy.verdict", "GREEDY IDENTICAL" if "GREEDY IDENTICAL" in text else "different", rel)

    # the quiet-host ABBA rerun
    vals = {}
    for arm in ("ref", "new"):
        for half in ("a", "b"):
            rel = f"{root}/2b-pp131k/{arm}-{half}/bench.json"
            b = bench(rel)
            vals[(arm, half)] = (b["summary"]["pp131072"]["tok_s_mean"], rel)
    for arm in ("ref", "new"):
        a, b = vals[(arm, "a")], vals[(arm, "b")]
        ds.add(f"abba.{arm}.a", a[0], f"{a[1]}#summary[pp131072].tok_s_mean", "tps")
        ds.add(f"abba.{arm}.b", b[0], f"{b[1]}#summary[pp131072].tok_s_mean", "tps")
        ds.add(f"abba.{arm}.mean", (a[0] + b[0]) / 2, "computed: mean of the two ABBA arms", "tps")
    ds.add(
        "abba.delta_pct",
        ((vals[("new", "a")][0] + vals[("new", "b")][0]) / 2 / ((vals[("ref", "a")][0] + vals[("ref", "b")][0]) / 2) - 1) * 100,
        "computed: ABBA means",
        "pct",
    )

    # the valid KLD + decode pair
    for arm, rel in [("ref", f"{root}/2-kv/ref1/bench.json"), ("new", f"{root}/2-kv/new2/bench.json")]:
        b = bench(rel)
        ds.add(f"pair.{arm}.pp32k", b["summary"]["pp32768"]["tok_s_mean"], f"{rel}#summary[pp32768].tok_s_mean", "tps")
        ds.add(f"pair.{arm}.pp131k", b["summary"]["pp131072"]["tok_s_mean"], f"{rel}#summary[pp131072].tok_s_mean", "tps")
        ds.add(f"pair.{arm}.kld", b["quality"]["kld_mean"], f"{rel}#quality.kld_mean", "kld")
        ds.add(f"pair.{arm}.top1", b["quality"]["top1_pct"], f"{rel}#quality.top1_pct", "pct")
        for test in ("conv", "struct", "prose"):
            ds.add(f"pair.{arm}.{test}", b["summary"][f"tg {test}"]["tok_s_mean"], f"{rel}#summary[tg {test}].tok_s_mean", "tps")

    # the release names, from the gate README
    gate_readme = f"{root}/README.md"
    gate_lines = read_lines(gate_readme)
    for n, line in enumerate(gate_lines, 1):
        joined = " ".join(gate_lines[n - 1 : n + 1])
        m = re.search(r"REF is the served sparse release `([0-9a-f]+)`.*NEW is the release tree of `d8-integration` at `([0-9a-f]+)`", joined)
        if m:
            ds.add("release.ref", m.group(1), f"{gate_readme} line {n}")
            ds.add("release.new", m.group(2), f"{gate_readme} line {n + 1}")
        if line.startswith("| 2 | KLD (100 chunks) and speed") and "PASS" in line:
            ds.add("gate2.rerun_verdict", "PASS", f"{gate_readme} line {n}")

    # spread across the three decode fixtures on the valid pair
    deltas = []
    for test in ("conv", "struct", "prose"):
        r = ds.values["pair"]["ref"][test]
        nw = ds.values["pair"]["new"][test]
        deltas.append(abs(nw / r - 1) * 100)
    ds.add("pair.decode_max_pct", max(deltas), "computed: the valid REF/NEW pair in this run", "pct")

    # sampling
    rel = f"{root}/3-sampling/compare.txt"
    for line in read_lines(rel):
        m = re.match(r"^(top1|top10|logprob|first_tv)\s+(\S+)?\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]\s+(\w+)", line)
        if m:
            name = m.group(1)
            ds.add(f"sampling.{name}.verdict", m.group(6), rel)
            ds.add(f"sampling.{name}.delta", float(m.group(3)), rel, "kld_delta")
            ds.add(f"sampling.{name}.ci", [float(m.group(4)), float(m.group(5))], rel, "kld_ci")
            if m.group(2):
                ds.add(f"sampling.{name}.a", float(m.group(2)), rel, "kld")

    # tools
    for name, rel in [("ref", f"{root}/4-tools/score-ref.txt"), ("new", f"{root}/4-tools/score-new.txt")]:
        for line in read_lines(rel):
            m = re.match(r"^(call|none|all)\s+(\d+)/(\d+)", line)
            if m:
                ds.add(f"tools.{name}.{m.group(1)}_ok", float(m.group(2)), rel, "int")
                ds.add(f"tools.{name}.{m.group(1)}_total", float(m.group(3)), rel, "int")
            m = re.match(r"agreement with the reference: (\d+)/(\d+)", line)
            if m:
                ds.add(f"tools.{name}.agree", float(m.group(1)), rel, "int")
    return ds


def rows_exhibit(rel: str, *, compressed: bool, label: str) -> tuple[dict, str]:
    if compressed:
        with lzma.open(ROOT / rel, "rt") as fh:
            lines = fh.read().splitlines()
    else:
        lines = read_lines(rel)
    chunks: dict[int, list] = {}
    for line in lines:
        if not line.strip():
            continue
        cells = line.split("\t")
        chunk = int(cells[0])
        chunks.setdefault(chunk, []).append(cells)
    if not chunks:
        raise LookupError(f"{rel}: no rows")
    means = {}
    for chunk, rows in chunks.items():
        means[chunk] = sum(float(r[5]) for r in rows) / len(rows)
    chosen = sorted(means, key=lambda c: means[c])[len(means) // 2]
    rows = chunks[chosen]
    positions = [int(r[1]) for r in rows]
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise LookupError(f"{rel}: positions are not contiguous")
    base_p = [math.exp(-float(r[4])) for r in rows]
    test_p = [math.exp(-float(r[3])) for r in rows]
    disagree = [int(r[6]) == 0 for r in rows]
    return {
        "label": label,
        "chunk": chosen,
        "first_position": positions[0],
        "last_position": positions[-1],
        "n_positions": len(rows),
        "mean_kld": means[chosen],
        "max_kld": max(float(r[5]) for r in rows),
        "disagree": sum(1 for d in disagree if d),
        "positions": positions,
        "base_p": base_p,
        "test_p": test_p,
        "disagree_flags": disagree,
    }, rel


def build_boundary() -> Dataset:
    ds = Dataset("boundary")
    for name, rel, compressed, label in [
        ("sparse", "evals/2026-11-17/halocline-kv-kld/rows/int8.tsv", False, "sparse int8"),
        ("dense", "evals/2026-11-17/halocline-dense-quality/q4kxl-int8-mmvq/rows.tsv.xz", True, "dense Q4_K_XL int8"),
    ]:
        ex, src = rows_exhibit(rel, compressed=compressed, label=label)
        for key in ("chunk", "first_position", "last_position", "n_positions", "disagree"):
            ds.add(f"{name}.{key}", ex[key], src, "tokens")
        ds.add(f"{name}.mean_kld", ex["mean_kld"], src, "kld")
        ds.add(f"{name}.max_kld", ex["max_kld"], src, "kld")
        # arrays are chart input, not display values; keep them out of the source map
        ds.values.setdefault(name, {})["base_p"] = ex["base_p"]
        ds.values.setdefault(name, {})["test_p"] = ex["test_p"]
        ds.values.setdefault(name, {})["disagree_flags"] = ex["disagree_flags"]
    return ds


def build_runs() -> Dataset:
    ds = Dataset("runs")
    rows = []
    # sparse quality and spec runs
    kv_dir = "evals/2026-11-17/halocline-kv-kld"
    for arm in ("int8", "fp16", "k8v4", "q4_0"):
        rel = f"{kv_dir}/{arm}/bench.json"
        b = bench(rel)
        rows.append(
            {
                "path": "sparse",
                "run": f"KV {arm}",
                "eval": f"{kv_dir}/",
                "date": "2026-11-17",
                "commit": b["commit"],
                "weights": "UD-IQ4_XS v3.1",
                "kv": arm,
                "spec": "8",
                "kld": b["quality"].get("kld_mean"),
                "p99": b["quality"].get("kld_p99"),
                "top1": b["quality"].get("top1_pct"),
                "pp": b["summary"].get("pp131072", {}).get("tok_s_mean"),
                "pp_depth": "131K",
                "decode": b["summary"].get("tg struct", {}).get("tok_s_mean"),
                "source": rel,
            }
        )
    # dense quality arms
    qdir = "evals/2026-11-17/halocline-dense-quality"
    for arm, label, weights, kv in [
        ("q4kxl-int8-mmvq", "D4 Q4_K_XL", "UD-Q4_K_XL", "int8"),
        ("iq4xs-int8", "D4 IQ4_XS MMQ", "UD-IQ4_XS", "int8"),
        ("iq4xs-int8-mmvq", "D4 IQ4_XS mmvq", "UD-IQ4_XS", "int8"),
    ]:
        rel = f"{qdir}/{arm}/bench.json"
        b = bench(rel)
        rows.append(
            {
                "path": "dense",
                "run": label,
                "eval": f"{qdir}/",
                "date": "2026-11-17",
                "commit": b["commit"],
                "weights": weights,
                "kv": kv,
                "spec": "8",
                "kld": b["quality"].get("kld_mean"),
                "p99": b["quality"].get("kld_p99"),
                "top1": b["quality"].get("top1_pct"),
                "pp": b["summary"].get("pp36864", {}).get("tok_s_mean"),
                "pp_depth": "36K",
                "decode": b["summary"].get("tg struct", {}).get("tok_s_mean"),
                "source": rel,
            }
        )
    # dense KV arms
    kdir = "evals/2026-11-17/halocline-d8-kv"
    for arm, label, kv in [
        ("int8", "D8 int8", "int8"),
        ("k8v4", "D8 K8V4", "k8v4"),
        ("k8v4-mtpint8", "D8 K8V4, drafter int8", "k8v4"),
        ("q4", "D8 rotated 4-bit (branch)", "q4_0"),
    ]:
        rel = f"{kdir}/{arm}/bench.json"
        b = bench(rel)
        rows.append(
            {
                "path": "dense",
                "run": label,
                "eval": f"{kdir}/",
                "date": "2026-11-17",
                "commit": b["commit"],
                "weights": "UD-Q4_K_XL",
                "kv": kv,
                "spec": "8",
                "kld": b["quality"].get("kld_mean"),
                "p99": b["quality"].get("kld_p99"),
                "top1": b["quality"].get("top1_pct"),
                "pp": b["summary"].get("pp36864", {}).get("tok_s_mean"),
                "pp_depth": "36K",
                "decode": b["summary"].get("tg struct", {}).get("tok_s_mean"),
                "source": rel,
            }
        )
    for i, row in enumerate(rows):
        src = row.pop("source")
        for key, fmt in [
            ("path", "text"), ("run", "text"), ("eval", "text"), ("date", "text"), ("commit", "text"),
            ("weights", "text"), ("kv", "text"), ("spec", "text"), ("kld", "kld"), ("p99", "kld"),
            ("top1", "pct"), ("pp", "tps"), ("pp_depth", "text"), ("decode", "tps"),
        ]:
            if row.get(key) is not None:
                ds.add(f"items.{i}.{key}", row[key], src + "#" + key, fmt)
    ds.add("count", len(rows), "computed: the bench.json files listed in this file", "int")
    return ds


BUILDERS = {
    "layers": build_layers,
    "machine": build_machine,
    "sparse": build_sparse,
    "dense": build_dense,
    "compare": build_compare,
    "regress": build_regress,
    "boundary": build_boundary,
    "runs": build_runs,
}


def build_all() -> dict[str, dict]:
    return {name: builder().as_json() for name, builder in BUILDERS.items()}


def write_all() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in build_all().items():
        path = DATA_DIR / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        written.append(path)
    return written


def check_all() -> int:
    bad = 0
    for name, data in build_all().items():
        path = DATA_DIR / f"{name}.json"
        if not path.exists():
            print(f"missing data file: {path.relative_to(ROOT)}")
            bad += 1
            continue
        have = json.loads(path.read_text())
        if have != data:
            print(f"stale data file: {path.relative_to(ROOT)} (run python3 site/extract.py --write)")
            bad += 1
    if bad:
        return 1
    print(f"data current: {len(BUILDERS)} files, {sum(len(b['_sources']) for b in build_all().values())} sourced values")
    return 0


def main(argv: list[str]) -> int:
    if "--write" in argv:
        for path in write_all():
            print(f"wrote {path.relative_to(ROOT)}")
        return 0
    if "--check" in argv:
        return check_all()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
