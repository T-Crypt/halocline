#!/usr/bin/env python3
"""Build the Halocline research site into site/_out.

Every number comes from site/data/*.json, which site/extract.py writes from the
eval files, logs and docs in this repository.  The build refuses to run when a
template types its own number, when a template or a claim asks for a key that is
not in the data, when a data file is stale, when a source file is missing, when
the output reaches another origin, or when a local link does not resolve.

No JavaScript framework, no CDN, no font service, no analytics.  The page reads
fully without JavaScript; the script only moves the depth gauge and fills the
chart tooltip.
"""

from __future__ import annotations

import html
import json
import math
import re
import shutil
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

import extract

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TEMPLATES = SITE / "templates"
STATIC = SITE / "static"
DATA = SITE / "data"
OUT = SITE / "_out"
ASSETS = ROOT / "assets"

SITE_URL = "https://t-crypt.github.io/halocline/"
REPO_URL = "https://github.com/T-Crypt/halocline"
ALLOWED_ORIGINS = (
    "https://t-crypt.github.io/halocline/",
    "https://github.com/T-Crypt/halocline",
    "https://huggingface.co/",
)
W3C_NAMESPACES = ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink", "http://www.w3.org/1999/xhtml")

PALETTE = {
    "accent": "#57ecdc",
    "halocline": "#57ecdc",
    "strata": "#e69f00",
    "ninfer": "#56b4e9",
    "llama": "#cc79a7",
    "no_drafter": "#86c4cc",
    "ink": "#cfe9e7",
    "muted": "#86c4cc",
    "faint": "#5f8f97",
    "grid": "#9ff3ff",
    "failure": "#ff8a5c",
}


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# values and sources


def source_href(source: str) -> str | None:
    if not source:
        return None
    if source.startswith("plan:") or source.startswith("site:"):
        return None
    if source.startswith("computed:"):
        return None
    m = re.match(r"^(evals/[^\s]+?|docs/[^\s]+?|README\.md|SPECS\.md|RESEARCH\.md|LICENSE|CONTRIBUTING\.md) line (\d+)$", source)
    if m:
        return f"{REPO_URL}/blob/main/{m.group(1)}#L{m.group(2)}"
    if source.startswith("evals/"):
        target = source.split("#")[0]
        target = target.split(" line ")[0]
        path = Path(target)
        suffix = path.suffix.lower()
        if suffix in (".json", ".md", ".txt", ".err", ".tsv", ".xz", ".sh", ".py", ".ids", ".jinja", ".sha256"):
            target = str(path.parent)
        return f"{REPO_URL}/tree/main/{target.lstrip('/')}"
    for prefix in ("docs/", "README.md", "SPECS.md", "RESEARCH.md", "LICENSE", "CONTRIBUTING.md"):
        if source.startswith(prefix):
            target = source
            if "#" in target:
                file, _, anchor = target.partition("#")
                return f"{REPO_URL}/blob/main/{file}#{anchor}"
            return f"{REPO_URL}/blob/main/{target}"
    return None


def source_label(source: str) -> str:
    if source.startswith("evals/"):
        return "eval"
    if source.startswith(("docs/", "README.md", "SPECS.md", "RESEARCH.md", "LICENSE", "CONTRIBUTING.md")):
        return "doc"
    if source.startswith("computed:"):
        return "computed"
    if source.startswith("plan:"):
        return "plan"
    return "source"


class Val:
    __slots__ = ("key", "value", "source", "fmt")

    def __init__(self, key: str, value, source: str, fmt: str):
        self.key = key
        self.value = value
        self.source = source
        self.fmt = fmt

    def __str__(self) -> str:
        return extract.format_value(self.value, self.fmt)

    @property
    def raw(self):
        return self.value

    @property
    def href(self) -> str | None:
        return source_href(self.source)

    @property
    def source_label(self) -> str:
        return source_label(self.source)


def flatten(name: str, data: dict, out: dict[str, Val]):
    values = data.get("values")
    sources = data.get("_sources") or {}
    formats = data.get("_formats") or {}
    if not isinstance(values, dict):
        raise BuildError(f"{name}: no values object")
    leaves = 0

    def walk(node, prefix=""):
        nonlocal leaves
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, f"{prefix}.{key}" if prefix else key)
            return
        if prefix.endswith((".base_p", ".test_p", ".disagree_flags")):
            return
        leaves += 1
        if prefix not in sources:
            raise BuildError(f"{name}: {prefix} has no source")
        if prefix not in formats:
            raise BuildError(f"{name}: {prefix} has no format")
        key = f"{name}.{prefix}"
        if key in out:
            raise BuildError(f"duplicate data key {key}")
        out[key] = Val(key, node, sources[prefix], formats[prefix])

    walk(values)
    if leaves != len(sources):
        missing = set(sources) - set(out)
        raise BuildError(f"{name}: {len(missing)} source entries with no value: {sorted(missing)[:5]}")


def load_data() -> dict[str, Val]:
    fresh = extract.build_all()
    data: dict[str, Val] = {}
    for name, expected in fresh.items():
        path = DATA / f"{name}.json"
        if not path.exists():
            raise BuildError(f"missing {path.relative_to(ROOT)}; run python3 site/extract.py --write")
        have = json.loads(path.read_text())
        if have != expected:
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            raise BuildError(f"stale {shown}; run python3 site/extract.py --write")
        flatten(name, have, data)
    return data


# ---------------------------------------------------------------------------
# small utilities


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def repo_dir(path: str) -> str:
    return f"{REPO_URL}/tree/main/{path.strip('/')}"


def repo_file(path: str, anchor: str | None = None) -> str:
    url = f"{REPO_URL}/blob/main/{path.lstrip('/')}"
    if anchor:
        url += f"#{anchor}"
    return url


def nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    frac = value / base
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if frac <= m:
            return m * base
    return 10 * base


def axis_tick(value: float) -> str:
    if value == 0:
        return "0"
    av = abs(value)
    if av < 0.01:
        return f"{value:.4f}"
    if av < 1:
        return f"{value:.3f}"
    if av < 1000:
        if abs(value - round(value)) < 0.05:
            return f"{value:,.0f}"
        return f"{value:.1f}"
    return f"{value:,.0f}"


def log_x(depth: float, lo: float = 4096, hi: float = 262144) -> float:
    return (math.log(max(depth, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))


def chart_svg_open(cid: str, title: str, desc: str, w: int, h: int) -> list[str]:
    return [
        f'<svg class="chart-svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="{cid}-title {cid}-desc">',
        f'<title id="{cid}-title">{esc(title)}</title>',
        f'<desc id="{cid}-desc">{esc(desc)}</desc>',
        '<g aria-hidden="true">',
    ]


def svg_text(x, y, text, size=11, fill=PALETTE["muted"], anchor="start", extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" {extra}>{esc(text)}</text>'
    )


class Chart:
    def __init__(self, cid, title, desc, svg, legend, headers, rows, caption):
        self.id = cid
        self.title = title
        self.desc = desc
        self.svg = svg
        self.legend = legend
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.sources = []


def line_chart(cid, title, desc, series, y_label, caption, x_ticks, y_fmt="tps"):
    w, h = 780, 390
    ml, mr, mt, mb = 70, 24, 24, 62
    pw, ph = w - ml - mr, h - mt - mb
    all_y = [p[1] for s in series for p in s["points"]]
    ymax = nice_max(max(all_y) * 1.08)
    ticks = [ymax * i / 4 for i in range(5)]
    parts = chart_svg_open(cid, title, desc, w, h)
    for t in ticks:
        y = mt + ph - (t / ymax) * ph
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-opacity="0.12"/>')
        parts.append(svg_text(ml - 8, y + 4, axis_tick(t), 10, PALETTE["faint"], "end"))
    for depth, label in x_ticks:
        x = ml + log_x(depth) * pw
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="{PALETTE["grid"]}" stroke-opacity="0.08"/>')
        parts.append(svg_text(x, mt + ph + 20, label, 10, PALETTE["faint"], "middle"))
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{w-mr}" y2="{mt+ph}" stroke="{PALETTE["grid"]}" stroke-opacity="0.35"/>')
    parts.append(svg_text(ml + pw / 2, h - 12, "context depth, tokens", 10, PALETTE["faint"], "middle"))
    parts.append(svg_text(16, mt + ph / 2, y_label, 10, PALETTE["faint"], "middle", f'transform="rotate(-90 16 {mt+ph/2:.1f})"'))
    for s in series:
        pts = []
        for depth, value in s["points"]:
            x = ml + log_x(depth) * pw
            y = mt + ph - (value / ymax) * ph
            pts.append((x, y, depth, value))
        if len(pts) > 1:
            d = " ".join(("M" if i == 0 else "L") + f"{x:.1f} {y:.1f}" for i, (x, y, _, _) in enumerate(pts))
            parts.append(
                f'<path class="line" pathLength="1" d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{s.get("width", 2.4)}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        for x, y, depth, value in pts:
            label = f'{s["name"]}: {extract.format_value(value, y_fmt)} at {extract.format_value(depth, "tokens")} tokens'
            parts.append(
                f'<g class="hover" data-label="{esc(label)}"><circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{s["color"]}" '
                f'stroke="#03141d" stroke-width="1.5"/><circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="transparent"/></g>'
            )
    parts.append("</g></svg>")
    legend = [{"name": s["name"], "color": s["color"]} for s in series]
    headers = ["Engine", "Cache", "Drafter", "Depth", "Value"]
    rows = []
    for s in series:
        for depth, value in s["points"]:
            rows.append([s["name"], s.get("cache", ""), s.get("drafter", ""), extract.format_value(depth, "tokens"), extract.format_value(value, y_fmt)])
    return Chart(cid, title, desc, "".join(parts), legend, headers, rows, caption)


def bar_chart(cid, title, desc, rows, x_label, caption, value_fmt="kld"):
    w = 780
    h = 92 + 42 * len(rows)
    ml, mr, mt = 238, 66, 30
    pw = w - ml - mr
    xmax = nice_max(max(r["value"] + (r.get("ci") or 0) for r in rows) * 1.08)
    parts = chart_svg_open(cid, title, desc, w, h)
    for i in range(5):
        v = xmax * i / 4
        x = ml + (v / xmax) * pw
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{h-40}" stroke="{PALETTE["grid"]}" stroke-opacity="0.12"/>')
        parts.append(svg_text(x, h - 22, axis_tick(v), 9, PALETTE["faint"], "middle"))
    for i, r in enumerate(rows):
        y = mt + i * 42
        color = r.get("color", PALETTE["accent"])
        parts.append(svg_text(8, y + 20, r["label"], 11, PALETTE["ink"], "start"))
        parts.append(f'<rect x="{ml}" y="{y+7}" width="{pw}" height="18" fill="{PALETTE["grid"]}" fill-opacity="0.04"/>')
        bw = (r["value"] / xmax) * pw
        parts.append(f'<rect x="{ml}" y="{y+7}" width="{bw:.1f}" height="18" fill="{color}" fill-opacity="0.85"/>')
        if r.get("ci"):
            x0 = ml + ((r["value"] - r["ci"]) / xmax) * pw
            x1 = ml + ((r["value"] + r["ci"]) / xmax) * pw
            parts.append(f'<line x1="{x0:.1f}" y1="{y+16}" x2="{x1:.1f}" y2="{y+16}" stroke="{PALETTE["ink"]}" stroke-width="1.4"/>')
            parts.append(f'<line x1="{x0:.1f}" y1="{y+12}" x2="{x0:.1f}" y2="{y+20}" stroke="{PALETTE["ink"]}" stroke-width="1.4"/>')
            parts.append(f'<line x1="{x1:.1f}" y1="{y+12}" x2="{x1:.1f}" y2="{y+20}" stroke="{PALETTE["ink"]}" stroke-width="1.4"/>')
        label = f'{r["label"]}: {extract.format_value(r["value"], value_fmt)}'
        if r.get("ci"):
            label += f' plus or minus {extract.format_value(r["ci"], value_fmt)}'
        parts.append(
            f'<g class="hover" data-label="{esc(label)}"><rect x="{ml}" y="{y}" width="{pw}" height="32" fill="transparent"/></g>'
        )
        parts.append(svg_text(ml + bw + 8, y + 20, extract.format_value(r["value"], value_fmt), 10, PALETTE["ink"], "start"))
    parts.append(svg_text(ml + pw / 2, h - 6, x_label, 10, PALETTE["faint"], "middle"))
    parts.append("</g></svg>")
    legend = []
    seen = set()
    for r in rows:
        c = r.get("color", PALETTE["accent"])
        if r.get("group") and c not in seen:
            legend.append({"name": r["group"], "color": c})
            seen.add(c)
    headers = ["Arm", "Value"]
    table_rows = [[r["label"], extract.format_value(r["value"], value_fmt)] for r in rows]
    return Chart(cid, title, desc, "".join(parts), legend, headers, table_rows, caption)


def interval_plot(cid, title, desc, rows, caption, value_fmt="kld_delta"):
    w = 780
    h = 95 + 40 * len(rows)
    ml, mr, mt = 238, 86, 32
    pw = w - ml - mr
    lo = min([0.0] + [r["lo"] for r in rows])
    hi = max([0.0] + [r["hi"] for r in rows])
    pad = (hi - lo) * 0.10
    lo -= pad
    hi += pad
    span = hi - lo

    def xpos(v):
        return ml + ((v - lo) / span) * pw

    parts = chart_svg_open(cid, title, desc, w, h)
    zero = xpos(0)
    parts.append(f'<line x1="{zero:.1f}" y1="{mt-8}" x2="{zero:.1f}" y2="{h-38}" stroke="{PALETTE["muted"]}" stroke-opacity="0.8" stroke-width="1.5"/>')
    parts.append(svg_text(zero, mt - 12, "zero", 9, PALETTE["muted"], "middle"))
    for i in range(5):
        v = lo + span * i / 4
        x = xpos(v)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{h-38}" stroke="{PALETTE["grid"]}" stroke-opacity="0.08"/>')
        parts.append(svg_text(x, h - 18, axis_tick(v), 9, PALETTE["faint"], "middle"))
    for i, r in enumerate(rows):
        y = mt + i * 40 + 14
        color = r.get("color", PALETTE["accent"])
        parts.append(svg_text(8, y + 4, r["label"], 11, PALETTE["ink"], "start"))
        x0, x1, xv = xpos(r["lo"]), xpos(r["hi"]), xpos(r["value"])
        parts.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x0:.1f}" y1="{y-5}" x2="{x0:.1f}" y2="{y+5}" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y-5}" x2="{x1:.1f}" y2="{y+5}" stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{xv:.1f}" cy="{y}" r="4.5" fill="{color}" stroke="#03141d" stroke-width="1.2"/>')
        parts.append(svg_text(x1 + 8, y + 4, extract.format_value(r["value"], value_fmt), 10, PALETTE["ink"], "start"))
        label = f'{r["label"]}: {extract.format_value(r["value"], value_fmt)}, interval {extract.format_value(r["lo"], value_fmt)} to {extract.format_value(r["hi"], value_fmt)}'
        parts.append(
            f'<g class="hover" data-label="{esc(label)}"><rect x="{ml}" y="{y-16}" width="{pw}" height="32" fill="transparent"/></g>'
        )
    parts.append(svg_text(ml + pw / 2, h - 4, "paired mean KLD difference", 10, PALETTE["faint"], "middle"))
    parts.append("</g></svg>")
    legend = []
    seen = set()
    for r in rows:
        c = r.get("color", PALETTE["accent"])
        if r.get("group") and c not in seen:
            legend.append({"name": r["group"], "color": c})
            seen.add(c)
    headers = ["Cache", "Paired difference", "Interval"]
    table_rows = [[r["label"], extract.format_value(r["value"], value_fmt), f'{extract.format_value(r["lo"], value_fmt)} to {extract.format_value(r["hi"], value_fmt)}'] for r in rows]
    return Chart(cid, title, desc, "".join(parts), legend, headers, table_rows, caption)


def overlap_chart(cid, title, desc, ex, caption):
    w, h = 780, 400
    ml, mr, mt, mb = 70, 24, 30, 66
    pw, ph = w - ml - mr, h - mt - mb
    positions = ex["positions"]
    base = ex["base_p"]
    test = ex["test_p"]
    flags = ex["disagree_flags"]
    ymin = -12.0
    xs = [ml + i / (len(positions) - 1) * pw for i in range(len(positions))]

    def ypos(p):
        lp = max(ymin, math.log10(max(p, 1e-12)))
        return mt + ph - ((lp - ymin) / (0 - ymin)) * ph

    parts = chart_svg_open(cid, title, desc, w, h)
    for exp in (0, -3, -6, -9, -12):
        y = mt + ph - ((exp - ymin) / (0 - ymin)) * ph
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-opacity="0.10"/>')
        label = "1" if exp == 0 else f"1e{exp}"
        parts.append(svg_text(ml - 8, y + 4, label, 10, PALETTE["faint"], "end"))
    area = [f"M {xs[0]:.1f} {ypos(base[0]):.1f}"]
    for i in range(1, len(xs)):
        area.append(f"L {xs[i]:.1f} {ypos(base[i]):.1f}")
    area.append(f"L {xs[-1]:.1f} {mt+ph:.1f} L {xs[0]:.1f} {mt+ph:.1f} Z")
    parts.append(f'<path d="{" ".join(area)}" fill="{PALETTE["no_drafter"]}" fill-opacity="0.22"/>')
    line = " ".join(("M" if i == 0 else "L") + f"{xs[i]:.1f} {ypos(test[i]):.1f}" for i in range(len(xs)))
    parts.append(f'<path class="line" pathLength="1" d="{line}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="2"/>')
    for i, flag in enumerate(flags):
        if flag:
            parts.append(f'<line x1="{xs[i]:.1f}" y1="{mt+ph+4}" x2="{xs[i]:.1f}" y2="{mt+ph+12}" stroke="{PALETTE["failure"]}" stroke-width="1.4"/>')
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{w-mr}" y2="{mt+ph}" stroke="{PALETTE["grid"]}" stroke-opacity="0.35"/>')
    first, last = positions[0], positions[-1]
    for frac, pos in ((0, first), (0.5, positions[len(positions)//2]), (1, last)):
        parts.append(svg_text(ml + frac * pw, mt + ph + 24, extract.format_value(pos, "tokens"), 10, PALETTE["faint"], "middle"))
    parts.append(svg_text(ml + pw / 2, h - 8, "scored position in the chunk", 10, PALETTE["faint"], "middle"))
    parts.append(svg_text(16, mt + ph / 2, "probability of the next token", 10, PALETTE["faint"], "middle", f'transform="rotate(-90 16 {mt+ph/2:.1f})"'))
    # one representative point to hover
    idx = len(positions) // 2
    label = (
        f'position {extract.format_value(positions[idx], "tokens")}: base {base[idx]:.4f}, engine {test[idx]:.4f}'
    )
    parts.append(
        f'<g class="hover" data-label="{esc(label)}"><circle cx="{xs[idx]:.1f}" cy="{ypos(test[idx]):.1f}" r="4" fill="{PALETTE["accent"]}"/>'
        f'<rect x="{xs[idx]-14:.1f}" y="{mt}" width="28" height="{ph}" fill="transparent"/></g>'
    )
    parts.append("</g></svg>")
    legend = [
        {"name": "Full-precision base", "color": PALETTE["no_drafter"]},
        {"name": "Halocline", "color": PALETTE["accent"]},
        {"name": "First choice differs", "color": PALETTE["failure"]},
    ]
    headers = ["Position", "Base probability", "Engine probability"]
    step = max(1, len(positions) // 10)
    table_rows = [
        [extract.format_value(positions[i], "tokens"), f"{base[i]:.5f}", f"{test[i]:.5f}"] for i in range(0, len(positions), step)
    ]
    return Chart(cid, title, desc, "".join(parts), legend, headers, table_rows, caption)


# ---------------------------------------------------------------------------
# the hero column


def hero_svg(v) -> str:
    w, h = 1440, 900
    boundary = int(h * 0.64)
    rng = [
        (76, 2.2, 0.9), (148, 1.6, 1.1), (226, 2.6, 0.8), (302, 1.4, 1.3),
        (388, 2.0, 0.95), (470, 1.8, 1.2), (552, 2.4, 0.85), (634, 1.5, 1.35),
        (716, 2.1, 1.0), (798, 1.7, 1.25), (882, 2.3, 0.9), (964, 1.6, 1.15),
        (1046, 2.0, 1.05), (1128, 1.4, 1.3), (1210, 2.5, 0.8), (1296, 1.8, 1.2),
    ]
    parts = [
        f'<svg class="hero-water" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">',
        "<defs>",
        '<linearGradient id="hero-upper" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#0f5566"/><stop offset="0.55" stop-color="#0a4357"/><stop offset="1" stop-color="#093345"/></linearGradient>',
        '<linearGradient id="hero-lower" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#05202d"/><stop offset="0.55" stop-color="#021019"/><stop offset="1" stop-color="#01070c"/></linearGradient>',
        '<radialGradient id="hero-caustic" cx="0.32" cy="0" r="0.9">'
        '<stop offset="0" stop-color="#a6f4ff" stop-opacity="0.18"/><stop offset="1" stop-color="#a6f4ff" stop-opacity="0"/></radialGradient>',
        '<radialGradient id="hero-pool" cx="0.5" cy="0.5" r="0.5">'
        '<stop offset="0" stop-color="#57ecdc" stop-opacity="0.24"/><stop offset="1" stop-color="#57ecdc" stop-opacity="0"/></radialGradient>',
        '<linearGradient id="hero-line" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#57ecdc" stop-opacity="0"/><stop offset="0.2" stop-color="#bafcf3" stop-opacity="0.9"/>'
        '<stop offset="0.5" stop-color="#f2fffd"/><stop offset="0.8" stop-color="#bafcf3" stop-opacity="0.9"/>'
        '<stop offset="1" stop-color="#57ecdc" stop-opacity="0"/></linearGradient>',
        '<linearGradient id="hero-shimmer" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#ffffff" stop-opacity="0"/><stop offset="0.5" stop-color="#ffffff" stop-opacity="0.8"/>'
        '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/></linearGradient>',
        '<pattern id="hero-grid" width="56" height="56" patternUnits="userSpaceOnUse">'
        '<path d="M56 0H0V56" fill="none" stroke="#9ff3ff" stroke-opacity="0.045" stroke-width="1"/></pattern>',
        '<filter id="hero-glow" x="-5%" y="-300%" width="110%" height="700%">'
        '<feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<clipPath id="hero-upper-clip"><rect x="0" y="0" width="' + str(w) + '" height="' + str(boundary) + '"/></clipPath>',
        "</defs>",
        f'<rect width="{w}" height="{h}" fill="url(#hero-lower)"/>',
        f'<rect width="{w}" height="{boundary}" fill="url(#hero-upper)"/>',
        f'<rect width="{w}" height="{boundary}" fill="url(#hero-caustic)"/>',
        f'<rect width="{w}" height="{h}" fill="url(#hero-grid)"/>',
        f'<g clip-path="url(#hero-upper-clip)" opacity="0.55">',
        '<path class="hero-caustic" d="M-80 116 C 240 78, 520 160, 820 104 S 1260 84, 1520 122" fill="none" stroke="#a6f4ff" stroke-opacity="0.12" stroke-width="24"/>',
        '<path class="hero-caustic second" d="M-80 280 C 300 236, 620 330, 940 268 S 1280 250, 1520 292" fill="none" stroke="#a6f4ff" stroke-opacity="0.08" stroke-width="32"/>',
        "</g>",
    ]
    for cx, r, dur in rng:
        rest = boundary - 12 - (cx % 23)
        delay = -((cx % 37) / 4)
        drift = (cx % 17) - 8
        parts.append(
            f'<circle class="hero-particle" cx="{cx}" cy="0" r="{r}" fill="#d8fffa" '
            f'style="--rest: {rest}px; --drift: {drift}px; --dur: {dur * 9:.1f}s; --delay: {delay:.1f}s"/>'
        )
    parts.extend(
        [
            f'<ellipse cx="{w*0.45:.0f}" cy="{boundary}" rx="{w*0.42:.0f}" ry="26" fill="url(#hero-pool)"/>',
            '<g fill="#e9fffb" opacity="0.5">'
            '<circle cx="148" cy="' + str(boundary - 5) + '" r="1.6"/><circle cx="246" cy="' + str(boundary - 3) + '" r="1.3"/>'
            '<circle cx="338" cy="' + str(boundary - 6) + '" r="1.9"/><circle cx="446" cy="' + str(boundary - 4) + '" r="1.2"/>'
            '<circle cx="556" cy="' + str(boundary - 5) + '" r="1.5"/><circle cx="658" cy="' + str(boundary - 6) + '" r="1.4"/>'
            '<circle cx="752" cy="' + str(boundary - 3) + '" r="1.8"/><circle cx="862" cy="' + str(boundary - 5) + '" r="1.2"/>'
            '<circle cx="964" cy="' + str(boundary - 6) + '" r="1.6"/><circle cx="1084" cy="' + str(boundary - 4) + '" r="1.7"/>'
            '<circle cx="1198" cy="' + str(boundary - 5) + '" r="1.3"/><circle cx="1312" cy="' + str(boundary - 6) + '" r="1.5"/>'
            "</g>",
            f'<g filter="url(#hero-glow)"><path class="hero-boundary" d="M0 {boundary} C 240 {boundary-4}, 480 {boundary+4}, 720 {boundary} '
            f'S 1200 {boundary-4}, 1440 {boundary}" fill="none" stroke="url(#hero-line)" stroke-width="2.6"/></g>',
            f'<rect class="hero-shimmer" x="-240" y="{boundary-3}" width="240" height="6" fill="url(#hero-shimmer)" opacity="0.55"/>',
            "</svg>",
        ]
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# page data


def build_charts(v) -> dict[str, Chart]:
    charts = {}
    x_ticks = [
        (v("layers.sunlit.depth").raw, str(v("layers.sunlit.label"))),
        (v("layers.twilight.depth").raw, str(v("layers.twilight.label"))),
        (v("layers.boundary.depth").raw, str(v("layers.boundary.label"))),
        (v("layers.deep.depth").raw, str(v("layers.deep.label"))),
        (v("layers.floor.depth").raw, str(v("layers.floor.label"))),
    ]
    decode_series = [
        {
            "name": "Halocline, no drafter",
            "color": PALETTE["no_drafter"],
            "cache": str(v("compare.sparse.halocline.kv")),
            "drafter": "none",
            "points": [
                (v("compare.sparse.no_drafter.36k.tokens").raw, v("compare.sparse.no_drafter.36k.decode").raw),
                (v("compare.sparse.no_drafter.137k.tokens").raw, v("compare.sparse.no_drafter.137k.decode").raw),
            ],
        },
        {
            "name": f"Strata {v('compare.sparse.strata.version')}, {v('compare.sparse.strata.drafter')}",
            "color": PALETTE["strata"],
            "cache": str(v("compare.sparse.strata.kv")),
            "drafter": str(v("compare.sparse.strata.drafter")),
            "points": [
                (v("compare.sparse.strata.36k.tokens").raw, v("compare.sparse.strata.36k.decode").raw),
                (v("compare.sparse.strata.137k.tokens").raw, v("compare.sparse.strata.137k.decode").raw),
            ],
        },
        {
            "name": "Halocline, MTP drafter",
            "color": PALETTE["accent"],
            "cache": str(v("compare.sparse.halocline.kv")),
            "drafter": "MTP",
            "width": 3,
            "points": [
                (v("compare.sparse.halocline.36k.tokens").raw, v("compare.sparse.halocline.36k.decode").raw),
                (v("compare.sparse.halocline.137k.tokens").raw, v("compare.sparse.halocline.137k.decode").raw),
            ],
        },
    ]
    charts["sparse_decode"] = line_chart(
        "sparse-decode",
        "Sparse decode against context depth",
        "Decode tokens a second at the two needle depths, for Halocline with and without its draft layer and for Strata with its own drafter. Lower is not better here: higher is faster.",
        decode_series,
        "decode tokens a second",
        "The line joins measured points; it is not a fitted model. Halocline's drafter raises decode at both depths. Strata's line is its quiet-host rerun.",
        x_ticks,
    )
    prompt_series = [
        {
            "name": "Halocline, no drafter",
            "color": PALETTE["no_drafter"],
            "points": [
                (v("compare.sparse.no_drafter.36k.tokens").raw, v("compare.sparse.no_drafter.36k.prompt").raw),
                (v("compare.sparse.no_drafter.137k.tokens").raw, v("compare.sparse.no_drafter.137k.prompt").raw),
            ],
        },
        {
            "name": f"Strata {v('compare.sparse.strata.version')}, {v('compare.sparse.strata.drafter')}",
            "color": PALETTE["strata"],
            "points": [
                (v("compare.sparse.strata.36k.tokens").raw, v("compare.sparse.strata.36k.prompt").raw),
                (v("compare.sparse.strata.137k.tokens").raw, v("compare.sparse.strata.137k.prompt").raw),
            ],
        },
        {
            "name": "Halocline, MTP drafter",
            "color": PALETTE["accent"],
            "width": 3,
            "points": [
                (v("compare.sparse.halocline.36k.tokens").raw, v("compare.sparse.halocline.36k.prompt").raw),
                (v("compare.sparse.halocline.137k.tokens").raw, v("compare.sparse.halocline.137k.prompt").raw),
            ],
        },
    ]
    charts["sparse_prompt"] = line_chart(
        "sparse-prompt",
        "Sparse prefill against context depth",
        "Prompt tokens a second at the same two needle depths. The vertical scale is thousands of tokens a second.",
        prompt_series,
        "prompt tokens a second",
        "The drafter takes VRAM from the expert cache, which is why its high-context prompt rate sits below the no-drafter serve.",
        x_ticks,
    )
    dense_series = [
        {
            "name": "NInfer",
            "color": PALETTE["ninfer"],
            "points": [
                (v("compare.dense.ninfer.tokens36k").raw, v("compare.dense.ninfer.pp36k").raw),
                (v("compare.dense.ninfer.tokens137k").raw, v("compare.dense.ninfer.pp137k").raw),
            ],
        },
        {
            "name": "Halocline dense",
            "color": PALETTE["accent"],
            "width": 3,
            "points": [
                (v("dense.depth16.tokens").raw, v("compare.dense.halocline.pp16k").raw),
                (v("dense.depth36.tokens").raw, v("compare.dense.halocline.pp36k").raw),
            ],
        },
    ]
    charts["dense_prefill"] = line_chart(
        "dense-prefill",
        "Dense prefill against context depth",
        "Prompt tokens a second for the dense model on the same card, against the NInfer floor. The arms do not share a protocol at every depth, so the caption names what is comparable.",
        dense_series,
        "prompt tokens a second",
        "Halocline's dense arm is the 8-bit cache gate at the first two depths; NInfer's smoke is at the two needle depths. The shared depth is the like-for-like reading.",
        x_ticks,
    )
    kv_rows = [
        ("fp16", PALETTE["no_drafter"]),
        ("k8v4", PALETTE["strata"]),
        ("q4_0", PALETTE["llama"]),
    ]
    interval_rows = []
    for arm, color in kv_rows:
        interval_rows.append(
            {
                "label": str(v(f"sparse.kv.arms.{arm}.flag")),
                "value": v(f"sparse.kv.arms.{arm}.delta").raw,
                "lo": v(f"sparse.kv.arms.{arm}.ci").raw[0],
                "hi": v(f"sparse.kv.arms.{arm}.ci").raw[1],
                "color": color,
            }
        )
    charts["sparse_kv"] = interval_plot(
        "sparse-kv",
        f"Each cache's distance from the {v('sparse.kv.arms.int8.label')} reference",
        f"The paired mean KLD difference against the {v('sparse.kv.arms.int8.label')} cache, with the gate's 95 percent interval. An interval that crosses zero cannot be ranked by this gate.",
        interval_rows,
        f"The {v('sparse.kv.arms.fp16.label')} cache's interval holds zero. The low-bit caches sit measurably further from the full-precision base.",
    )
    dense_rows = [
        {
            "label": f"Halocline, {v('dense.packs.q4kxl.label')}, {v('dense.kv.arms.int8.label')}",
            "value": v("dense.quality.arms.halo_q4kxl_int8.quality.kld_mean").raw,
            "color": PALETTE["accent"],
            "group": "Halocline",
        },
        {
            "label": f"llama.cpp, {v('dense.packs.q4kxl.label')}, {v('dense.context.fp16.label')}",
            "value": v("dense.quality.arms.llama_q4kxl_f16.mean_kld").raw,
            "ci": v("dense.quality.arms.llama_q4kxl_f16.ci").raw[1],
            "color": PALETTE["llama"],
            "group": "llama.cpp",
        },
        {
            "label": f"Halocline, {v('dense.packs.iq4xs.label')}, MMQ",
            "value": v("dense.quality.arms.halo_iq4xs_int8.quality.kld_mean").raw,
            "color": PALETTE["accent"],
            "group": "Halocline",
        },
        {
            "label": f"Halocline, {v('dense.packs.iq4xs.label')}, mmvq",
            "value": v("dense.quality.arms.halo_iq4xs_mmvq.quality.kld_mean").raw,
            "color": PALETTE["accent"],
            "group": "Halocline",
        },
        {
            "label": f"llama.cpp, {v('dense.packs.iq4xs.label')}, {v('dense.context.fp16.label')}",
            "value": v("dense.quality.arms.llama_iq4xs_f16.mean_kld").raw,
            "ci": v("dense.quality.arms.llama_iq4xs_f16.ci").raw[1],
            "color": PALETTE["llama"],
            "group": "llama.cpp",
        },
        {
            "label": f"llama.cpp, {v('dense.packs.iq4xs.label')}, {v('dense.quality.arms.llama_iq4xs_q8.kv')}",
            "value": v("dense.quality.arms.llama_iq4xs_q8.mean_kld").raw,
            "ci": v("dense.quality.arms.llama_iq4xs_q8.ci").raw[1],
            "color": PALETTE["llama"],
            "group": "llama.cpp",
        },
    ]
    charts["dense_kld"] = bar_chart(
        "dense-kld",
        "Dense mean KLD against the full-precision base",
        f"Mean KLD over the {v('dense.kv.arms.int8.quality.chunks')}-chunk gate; lower is closer to the base. Halocline and llama.cpp are on the same GGUF for each quant; the whiskers are llama.cpp's own error bar.",
        dense_rows,
        "mean KLD, lower is better",
        f"{v('dense.packs.q4kxl.label')} sits {v('dense.quality.ratio')} closer to the base than {v('dense.packs.iq4xs.label')}. Halocline's {v('dense.packs.q4kxl.label')} arm sits inside llama.cpp's error bar on the same weights.",
    )
    b = json.loads((DATA / "boundary.json").read_text())["values"]["sparse"]
    ex = {
        "positions": list(range(int(b["first_position"]), int(b["last_position"]) + 1)),
        "base_p": b["base_p"],
        "test_p": b["test_p"],
        "disagree_flags": b["disagree_flags"],
    }
    charts["boundary_sparse"] = overlap_chart(
        "boundary-sparse",
        "One sparse chunk at the boundary",
        "The full-precision base's probability of the ground-truth next token, and the engine's, at every scored position of one chunk. Ticks below the axis mark positions where the first choice differs.",
        ex,
        f"Chunk {v('boundary.sparse.chunk')}, positions {v('boundary.sparse.first_position')} to {v('boundary.sparse.last_position')}, mean KLD {v('boundary.sparse.mean_kld')}. The rows file stores the target token's log-probability, not the full vocabulary, so this is the overlap the data supports.",
    )
    bd = json.loads((DATA / "boundary.json").read_text())["values"]["dense"]
    exd = {
        "positions": list(range(int(bd["first_position"]), int(bd["last_position"]) + 1)),
        "base_p": bd["base_p"],
        "test_p": bd["test_p"],
        "disagree_flags": bd["disagree_flags"],
    }
    charts["boundary_dense"] = overlap_chart(
        "boundary-dense",
        "One dense chunk at the boundary",
        "The base's probability of the ground-truth next token, and the dense engine's, at every scored position of one chunk. Ticks below the axis mark positions where the first choice differs.",
        exd,
        f"Chunk {v('boundary.dense.chunk')}, positions {v('boundary.dense.first_position')} to {v('boundary.dense.last_position')}, mean KLD {v('boundary.dense.mean_kld')}. The rows file stores the target token's log-probability, not the full vocabulary.",
    )
    srcs = {
        "sparse_decode": ["evals/2026-11-17/halocline-llama-swap/", "evals/2026-11-17/halocline-sparse-serve/", "evals/2026-11-16/halocline-p0-floors/"],
        "sparse_prompt": ["evals/2026-11-17/halocline-llama-swap/", "evals/2026-11-17/halocline-sparse-serve/", "evals/2026-11-16/halocline-p0-floors/"],
        "dense_prefill": ["evals/2026-11-17/halocline-d8-kv/", "evals/2026-11-16/halocline-p0-floors/"],
        "sparse_kv": ["evals/2026-11-17/halocline-kv-kld/"],
        "dense_kld": ["evals/2026-11-17/halocline-dense-quality/"],
        "boundary_sparse": ["evals/2026-11-17/halocline-kv-kld/"],
        "boundary_dense": ["evals/2026-11-17/halocline-dense-quality/"],
    }
    for cid, paths in srcs.items():
        charts[cid].sources = [{"href": repo_dir(p), "label": "eval"} for p in paths]
    return charts


def build_tables(v) -> dict[str, list[dict]]:
    tables = {}
    tables["sparse_speed"] = [
        {
            "name": "Halocline, MTP",
            "cache": str(v("compare.sparse.halocline.kv")),
            "warm": f"{v('compare.sparse.halocline.warm_low')} to {v('compare.sparse.halocline.warm_high')}",
            "d36": v("compare.sparse.halocline.36k.decode"),
            "p36": v("compare.sparse.halocline.36k.prompt"),
            "d137": v("compare.sparse.halocline.137k.decode"),
            "p137": v("compare.sparse.halocline.137k.prompt"),
            "href": repo_dir("evals/2026-11-17/halocline-llama-swap/"),
        },
        {
            "name": "Halocline, no drafter",
            "cache": str(v("compare.sparse.halocline.kv")),
            "warm": f"{v('compare.sparse.no_drafter.warm_low')} to {v('compare.sparse.no_drafter.warm_high')}",
            "d36": v("compare.sparse.no_drafter.36k.decode"),
            "p36": v("compare.sparse.no_drafter.36k.prompt"),
            "d137": v("compare.sparse.no_drafter.137k.decode"),
            "p137": v("compare.sparse.no_drafter.137k.prompt"),
            "href": repo_dir("evals/2026-11-17/halocline-sparse-serve/"),
        },
        {
            "name": f"Strata {v('compare.sparse.strata.version')}, {v('compare.sparse.strata.drafter')}",
            "cache": str(v("compare.sparse.strata.kv")),
            "warm": f"{v('compare.sparse.strata.warm_low')} to {v('compare.sparse.strata.warm_high')}",
            "d36": v("compare.sparse.strata.36k.decode"),
            "p36": v("compare.sparse.strata.36k.prompt"),
            "d137": v("compare.sparse.strata.137k.decode"),
            "p137": v("compare.sparse.strata.137k.prompt"),
            "href": repo_dir("evals/2026-11-16/halocline-p0-floors/"),
        },
    ]
    tables["dense_speed"] = [
        {
            "name": "Halocline dense, int8 cache",
            "pp16": v("compare.dense.halocline.pp16k"),
            "pp36": v("compare.dense.halocline.pp36k"),
            "pp137": "not measured",
            "d36": v("compare.dense.halocline.conv"),
            "href": repo_dir("evals/2026-11-17/halocline-d8-kv/"),
        },
        {
            "name": "NInfer, its own format",
            "pp16": "not measured",
            "pp36": v("compare.dense.ninfer.pp36k"),
            "pp137": v("compare.dense.ninfer.pp137k"),
            "d36": v("compare.dense.ninfer.decode36k"),
            "href": repo_dir("evals/2026-11-16/halocline-p0-floors/"),
        },
    ]
    kv_rows = []
    for arm in ("int8", "fp16", "k8v4", "q4_0"):
        kv_rows.append(
            {
                "label": v(f"sparse.kv.arms.{arm}.flag"),
                "kld": v(f"sparse.kv.arms.{arm}.quality.kld_mean"),
                "top1": v(f"sparse.kv.arms.{arm}.quality.top1_pct"),
                "delta": v(f"sparse.kv.arms.{arm}.delta") if arm != "int8" else "reference",
                "ci": v(f"sparse.kv.arms.{arm}.ci") if arm != "int8" else "\u2014",
                "pp32": v(f"sparse.kv.arms.{arm}.speed.pp32768"),
                "pp131": v(f"sparse.kv.arms.{arm}.speed.pp131072"),
                "decode": v(f"sparse.kv.arms.{arm}.speed.tg_struct"),
                "href": repo_dir("evals/2026-11-17/halocline-kv-kld/"),
            }
        )
    tables["kv_sparse"] = kv_rows
    draft_rows = []
    for spec in ("5", "8", "12"):
        draft_rows.append(
            {
                "label": v(f"sparse.draft.arms.{spec}.label"),
                "conv": v(f"sparse.draft.arms.{spec}.speed.tg_conv"),
                "struct": v(f"sparse.draft.arms.{spec}.speed.tg_struct"),
                "prose": v(f"sparse.draft.arms.{spec}.speed.tg_prose"),
                "accept": v(f"sparse.draft.arms.{spec}.accept.tg_struct"),
                "tpp": v(f"sparse.draft.arms.{spec}.tokens_per_pass"),
                "href": repo_dir("evals/2026-11-17/halocline-kv-kld/"),
            }
        )
    tables["draft"] = draft_rows
    dense_q = []
    for key in ("halo_q4kxl_int8", "llama_q4kxl_f16", "halo_iq4xs_int8", "halo_iq4xs_mmvq", "llama_iq4xs_f16", "llama_iq4xs_q8"):
        dense_q.append(
            {
                "label": v(f"dense.quality.arms.{key}.label").raw,
                "kld": v(f"dense.quality.arms.{key}.quality.kld_mean") if key.startswith("halo") else v(f"dense.quality.arms.{key}.mean_kld"),
                "err": v(f"dense.quality.arms.{key}.err") if key.startswith("llama") else "not reported",
                "top1": v(f"dense.quality.arms.{key}.quality.top1_pct") if key.startswith("halo") else v(f"dense.quality.arms.{key}.top1_pct"),
                "ppl": v(f"dense.quality.arms.{key}.quality.ppl") if key.startswith("halo") else v(f"dense.quality.arms.{key}.mean_ppl"),
                "href": repo_dir("evals/2026-11-17/halocline-dense-quality/"),
            }
        )
    tables["dense_quality"] = dense_q
    dense_kv = []
    for key in ("int8", "k8v4", "k8v4-mtpint8", "q4"):
        slug = "q4" if key == "q4" else ("k8v4" if key.startswith("k8v4") else "int8")
        dense_kv.append(
            {
                "label": v(f"dense.kv.arms.{key}.label"),
                "cell": v(f"dense.context.{slug}.cell_bytes"),
                "kld": v(f"dense.kv.arms.{key}.quality.kld_mean"),
                "top1": v(f"dense.kv.arms.{key}.quality.top1_pct"),
                "delta": v(f"dense.kv.arms.{key}.delta") if key != "int8" else "reference",
                "ci": v(f"dense.kv.arms.{key}.ci") if key != "int8" else "\u2014",
                "href": repo_dir("evals/2026-11-17/halocline-d8-kv/"),
            }
        )
    tables["kv_dense"] = dense_kv

    def maybe(key):
        try:
            return v(key)
        except BuildError:
            return None

    tables["context"] = [
        {
            "label": v(f"dense.context.{slug}.label"),
            "cell": v(f"dense.context.{slug}.cell_bytes"),
            "first": v(f"dense.context.{slug}.max_first"),
            "mtp": maybe(f"dense.context.{slug}.max_mtp"),
            "ring": maybe(f"dense.context.{slug}.max_ring"),
            "href": repo_file("docs/precision/qwen38-27b.md"),
        }
        for slug in ("fp16", "int8", "k8v4", "q4")
    ]
    tables["prefill"] = []
    for arm in ("int8", "k8v4"):
        row = {
            "name": v(f"dense.kv.arms.{arm}.label"),
            "pp16a": v(f"dense.prefill.{arm}.pp16k_a"),
            "pp16b": v(f"dense.prefill.{arm}.pp16k_b"),
            "pp16m": v(f"dense.prefill.{arm}.pp16k_mean"),
            "pp36a": v(f"dense.prefill.{arm}.pp36k_a"),
            "pp36b": v(f"dense.prefill.{arm}.pp36k_b"),
            "pp36m": v(f"dense.prefill.{arm}.pp36k_mean"),
            "href": repo_dir("evals/2026-11-17/halocline-d8-kv/"),
        }
        if arm == "k8v4":
            row["d16"] = v("dense.prefill.delta_16k_pct")
            row["d36"] = v("dense.prefill.delta_36k_pct")
        else:
            row["d16"] = "reference"
            row["d36"] = "reference"
        tables["prefill"].append(row)
    tables["runs_sparse"] = []
    tables["runs_dense"] = []
    for i in range(v("runs.count").raw):
        row = {
            "run": v(f"runs.items.{i}.run"),
            "eval": v(f"runs.items.{i}.eval"),
            "commit": v(f"runs.items.{i}.commit"),
            "weights": v(f"runs.items.{i}.weights"),
            "kv": v(f"runs.items.{i}.kv"),
            "kld": v(f"runs.items.{i}.kld"),
            "p99": v(f"runs.items.{i}.p99"),
            "top1": v(f"runs.items.{i}.top1"),
            "pp": v(f"runs.items.{i}.pp"),
            "pp_depth": v(f"runs.items.{i}.pp_depth"),
            "decode": v(f"runs.items.{i}.decode"),
            "href": repo_dir(str(v(f"runs.items.{i}.eval"))),
        }
        tables["runs_sparse" if str(v(f"runs.items.{i}.path")) == "sparse" else "runs_dense"].append(row)

    tables["gates"] = [
        {
            "gate": "Greedy identity",
            "verdict": v("regress.gate1.verdict"),
            "note": f"{v('regress.greedy.ref_tokens')} of {v('regress.greedy.new_tokens')} tokens identical",
            "href": repo_dir("evals/2026-11-17/halocline-sparse-regress/"),
        },
        {
            "gate": "KLD and speed",
            "verdict": f"{v('regress.gate2.saved_verdict')}, then {v('regress.gate2.rerun_verdict')} on the quiet-host rerun",
            "note": f"KLD {v('regress.pair.new.kld')}, first choice {v('regress.pair.new.top1')}; high-context prefill {v('regress.abba.delta_pct')} on ABBA means",
            "href": repo_dir("evals/2026-11-17/halocline-sparse-regress/"),
        },
        {
            "gate": "Sampling distribution",
            "verdict": v("regress.gate3.verdict"),
            "note": "first choice, first ten, log-probability and first-token TV inside the reference spread",
            "href": repo_dir("evals/2026-11-17/halocline-sparse-regress/"),
        },
        {
            "gate": "Tool calls",
            "verdict": v("regress.gate4.verdict"),
            "note": f"{v('regress.tools.new.agree')} of {v('regress.tools.new.all_total')} cases agree with the reference",
            "href": repo_dir("evals/2026-11-17/halocline-sparse-regress/"),
        },
    ]
    return tables


# ---------------------------------------------------------------------------
# content


PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")

DOC_LABELS = (
    ("docs/roadmap.md", "the roadmap"),
    ("docs/testing.md", "the gates"),
    ("docs/community/", "the community row"),
    ("docs/precision/", "the precision page"),
    ("docs/research/", "the research notes"),
    ("README.md", "the README"),
    ("SPECS.md", "the reference machine"),
    ("CONTRIBUTING.md", "contributing"),
    ("LICENSE", "the licence"),
)


def doc_label(path: str) -> str:
    for prefix, label in DOC_LABELS:
        if path.startswith(prefix):
            return label
    return path


def resolve(template_text: str, values: dict, v, where: str) -> str:
    used = []

    def repl(m):
        name = m.group(1)
        if name not in values:
            raise BuildError(f"{where}: placeholder {{{name}}} has no value key")
        used.append(name)
        return str(v(values[name]))

    text = PLACEHOLDER.sub(repl, template_text)
    if "{" in text or "}" in text:
        raise BuildError(f"{where}: unresolved brace in {text!r}")
    for name, key in values.items():
        try:
            v(key)
        except BuildError as exc:
            raise BuildError(f"{where}: {name} -> {exc}") from None
    return text


def sources_for(values: dict, v) -> list[dict]:
    seen = set()
    out = []
    for key in values.values():
        val = v(key)
        if val.href and val.href not in seen:
            seen.add(val.href)
            out.append({"href": val.href, "label": val.source_label})
    return out


def load_content(v):
    findings_raw = json.loads((SITE / "content" / "findings.json").read_text())["findings"]
    findings = []
    for f in findings_raw:
        where = f"finding {f['id']}"
        f = dict(f)
        f["title"] = resolve(f["title"], f.get("values", {}), v, where)
        f["claim"] = resolve(f["claim"], f.get("values", {}), v, where)
        f["caveat"] = resolve(f["caveat"], f.get("values", {}), v, where)
        f["eval_links"] = [{"href": repo_dir(p), "label": p} for p in f.get("evals", [])]
        f["doc_links"] = [{"href": repo_file(p.split("#")[0], p.split("#")[1] if "#" in p else None), "label": doc_label(p)} for p in f.get("docs", [])]
        f["sources"] = sources_for(f.get("values", {}), v)
        f["status_class"] = {"measured": "measured", "open": "open", "on a branch": "branch"}.get(f["status"], "open")
        findings.append(f)
    round_raw = json.loads((SITE / "content" / "roundings.json").read_text())["rows"]
    roundings = []
    for r in round_raw:
        where = f"rounding {r['part']}/{r['rounding']}"
        r = dict(r)
        r["rounding"] = resolve(r["rounding"], r.get("values", {}), v, where)
        r["when"] = resolve(r["when"], r.get("values", {}), v, where)
        r["cost"] = resolve(r["cost"], r.get("values", {}), v, where)
        r["eval_links"] = [{"href": repo_dir(p), "label": p} for p in r.get("evals", [])]
        r["doc_links"] = [{"href": repo_file(p.split("#")[0], p.split("#")[1] if "#" in p else None), "label": doc_label(p)} for p in r.get("docs", [])]
        r["sources"] = sources_for(r.get("values", {}), v)
        r["status_class"] = {"measured": "measured", "open": "open", "on a branch": "branch", "exact": "exact", "interim": "interim", "not isolated": "not-isolated"}.get(r["status"], "open")
        roundings.append(r)
    open_raw = json.loads((SITE / "content" / "open.json").read_text())["items"]
    open_items = []
    for item in open_raw:
        item = dict(item)
        item["doc_links"] = [{"href": repo_file(p.split("#")[0], p.split("#")[1] if "#" in p else None), "label": doc_label(p)} for p in item.get("docs", [])]
        open_items.append(item)
    return findings, roundings, open_items


# ---------------------------------------------------------------------------
# checks


def template_literals(template_dir: Path) -> list[str]:
    problems = []
    for path in sorted(template_dir.rglob("*.html")):
        text = path.read_text()
        text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)
        text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.S)
        text = re.sub(r"\{%.*?%\}", "", text, flags=re.S)
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        text = re.sub(r"<(style|script)\b.*?</\1>", "", text, flags=re.S | re.I)
        attrs = re.findall(r'\b(?:alt|aria-label|title)="([^"]*)"', text)
        text = re.sub(r"<[^>]*>", "", text)
        for blob, where in [(text, "text"), *[(a, "attribute") for a in attrs]]:
            for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", blob):
                line = text[: m.start()].count("\n") + 1 if where == "text" else 0
                snippet = blob[max(0, m.start() - 35) : m.end() + 35].replace("\n", " ")
                try:
                    shown = path.relative_to(ROOT)
                except ValueError:
                    shown = path
                problems.append(f"{shown}:{line}: {where} has a literal number: ...{snippet}...")
    return problems


def content_problems(findings: list[dict], roundings: list[dict], open_items: list[dict]) -> list[str]:
    problems = []
    for f in findings:
        for key in ("title", "claim", "caveat"):
            text = PLACEHOLDER.sub("", f.get(key, ""))
            if re.search(r"\d", text):
                problems.append(f"finding {f['id']}: {key} has a literal digit: {text!r}")
    for r in roundings:
        for key in ("rounding", "when", "cost"):
            text = PLACEHOLDER.sub("", r.get(key, ""))
            if re.search(r"\d", text):
                problems.append(f"rounding {r['part']}: {key} has a literal digit: {text!r}")
    for item in open_items:
        for key in ("title", "status", "accept"):
            text = item.get(key, "")
            if re.search(r"\d", text):
                problems.append(f"open item {item['title']}: {key} has a literal digit: {text!r}")
    return problems


def content_literals() -> list[str]:
    findings = json.loads((SITE / "content" / "findings.json").read_text())["findings"]
    roundings = json.loads((SITE / "content" / "roundings.json").read_text())["rows"]
    open_items = json.loads((SITE / "content" / "open.json").read_text())["items"]
    return content_problems(findings, roundings, open_items)


def source_problems() -> list[str]:
    problems = []
    for path in sorted(DATA.glob("*.json")):
        data = json.loads(path.read_text())
        sources = data.get("_sources") or {}
        for key, source in sources.items():
            if source.startswith(("plan:", "site:", "computed:")):
                continue
            m = re.match(r"^(.*?) line (\d+)$", source)
            target = m.group(1) if m else source.split("#")[0]
            file = ROOT / target
            if not file.exists():
                problems.append(f"{path.name}: {key}: source file does not exist: {target}")
                continue
            if m:
                count = len(file.read_text().splitlines())
                if int(m.group(2)) > count:
                    problems.append(f"{path.name}: {key}: line {m.group(2)} past the end of {target} ({count} lines)")
    return problems


def check_origins(out: Path) -> list[str]:
    problems = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".html", ".css", ".js", ".svg"):
            continue
        text = path.read_text(errors="replace")
        for m in re.finditer(r"https?://[^\s\"'<>)]+", text):
            url = m.group(0).rstrip(".,;")
            if url.startswith(W3C_NAMESPACES):
                continue
            if not url.startswith(ALLOWED_ORIGINS):
                problems.append(f"{path.relative_to(out)}: reaches {url}")
        for m in re.finditer(r'(?:src|href)="(//[^"]+)"', text):
            problems.append(f"{path.relative_to(out)}: protocol-relative URL {m.group(1)}")
    return problems


def check_links(out: Path) -> list[str]:
    problems = []
    ids = {}
    for path in out.rglob("*.html"):
        ids[path.resolve()] = set(re.findall(r'\bid="([^"]+)"', path.read_text()))
    for path in sorted(out.rglob("*.html")):
        text = path.read_text()
        for attr in ("href", "src"):
            for m in re.finditer(rf'\b{attr}="([^"]+)"', text):
                url = html.unescape(m.group(1))
                if url.startswith(("http://", "https://", "mailto:", "tel:", "data:")):
                    continue
                target, _, frag = url.partition("#")
                if not target:
                    if frag and frag not in ids.get(path.resolve(), set()):
                        problems.append(f"{path.relative_to(out)}: missing anchor #{frag}")
                    continue
                target = target.split("?")[0]
                resolved = (path.parent / target).resolve()
                if resolved.is_dir():
                    resolved = resolved / "index.html"
                if not resolved.exists():
                    problems.append(f"{path.relative_to(out)}: broken link {url}")
                elif frag and resolved.suffix == ".html":
                    if frag not in ids.get(resolved.resolve(), set()):
                        problems.append(f"{path.relative_to(out)}: missing anchor {url}")
    return problems


def check_weight(out: Path) -> list[tuple[str, int]]:
    weights = []
    for page in sorted(out.rglob("*.html")):
        total = page.stat().st_size
        seen = set()
        for m in re.finditer(r'<(?:img|script|link|source)\b[^>]*\b(?:src|href|srcset)="([^"]+)"', page.read_text()):
            rel = m.group(1).split()[0].rstrip(",")
            if rel.startswith(("http://", "https://", "#")):
                continue
            asset = (page.parent / rel).resolve()
            if asset in seen or not asset.exists() or asset.is_dir():
                continue
            seen.add(asset)
            total += asset.stat().st_size
        weights.append((str(page.relative_to(out)), total))
    return weights


# ---------------------------------------------------------------------------
# pages


def base_context(v, charts, tables, findings, roundings, open_items):
    return {
        "v": v,
        "charts": charts,
        "tables": tables,
        "findings": findings,
        "roundings": roundings,
        "open_items": open_items,
        "site_url": SITE_URL,
        "repo_url": REPO_URL,
        "palette": PALETTE,
        "hero_svg": hero_svg(v),
        "layers": [
            {
                "depth": v(f"layers.{slug}.depth"),
                "label": v(f"layers.{slug}.label"),
                "name": v(f"layers.{slug}.name"),
            }
            for slug in ("surface", "sunlit", "twilight", "boundary", "deep", "floor")
        ],
    }


def render(env: Environment, template: str, page: dict, ctx: dict) -> str:
    return env.get_template(template).render(page=page, **ctx)


def page(title: str, description: str, canonical: str, prefix: str, body_class: str, nav_current: str):
    return {
        "title": title,
        "description": description,
        "canonical": canonical,
        "og_image": SITE_URL + "assets/halocline-social-preview.png",
        "prefix": prefix,
        "body_class": body_class,
        "nav_current": nav_current,
    }


def resolver(data: dict[str, Val]):
    def v(key: str) -> Val:
        try:
            return data[key]
        except KeyError:
            raise BuildError(f"no data key {key!r}") from None

    return v


def build() -> int:
    data = load_data()
    v = resolver(data)
    before = template_literals(TEMPLATES)
    if before:
        raise BuildError("a template types its own number:\n  " + "\n  ".join(before))
    content_before = content_literals()
    if content_before:
        raise BuildError("content types its own number:\n  " + "\n  ".join(content_before))
    src_before = source_problems()
    if src_before:
        raise BuildError("a data value names a source that is not there:\n  " + "\n  ".join(src_before))
    charts = build_charts(v)
    tables = build_tables(v)
    findings, roundings, open_items = load_content(v)
    ctx = base_context(v, charts, tables, findings, roundings, open_items)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(v=v, repo_dir=repo_dir, repo_file=repo_file, repo_url=REPO_URL, site_url=SITE_URL)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(STATIC, OUT / "static")
    shutil.copytree(ASSETS, OUT / "assets")

    pages = [
        (
            "index.html",
            page(
                "Halocline: the engine that hides nothing",
                "A research record of one inference engine on one Ada card: what was measured, how, and what is still open.",
                SITE_URL,
                "",
                "page-index",
                "dive",
            ),
            "index.html",
        ),
        (
            "models/sparse.html",
            page(
                f"Sparse path: {v('machine.models.sparse.name')}",
                "The sparse model's weights, expert placement, quality against the full-precision base, and speed on one card.",
                SITE_URL + "models/sparse.html",
                "../",
                "page-model",
                "sparse",
            ),
            "sparse.html",
        ),
        (
            "models/dense.html",
            page(
                f"Dense path: {v('machine.models.dense.name')}",
                "The dense model's packs, lossless load, KLD against the same base, cache formats, context ceiling and speed.",
                SITE_URL + "models/dense.html",
                "../",
                "page-model",
                "dense",
            ),
            "dense.html",
        ),
    ]
    (OUT / "models").mkdir(exist_ok=True)
    for rel, page_ctx, template in pages:
        rendered = render(env, template, page_ctx, ctx)
        (OUT / rel).write_text(rendered)

    origin_before = check_origins(OUT)
    if origin_before:
        raise BuildError("the output reaches another origin:\n  " + "\n  ".join(origin_before))
    link_before = check_links(OUT)
    if link_before:
        raise BuildError("a local link does not resolve:\n  " + "\n  ".join(link_before))
    weights = check_weight(OUT)
    heavy = [(name, size) for name, size in weights if size > 400 * 1024]
    if heavy:
        raise BuildError("a page is over the 400 KB budget before fonts:\n  " + "\n  ".join(f"{n}: {s} bytes" for n, s in heavy))

    values = len(data)
    print(f"built {len(pages)} pages from {values} sourced values")
    for name, size in weights:
        print(f"  {name}: {size / 1024:.1f} KB")
    print(f"  data files: {len(list(DATA.glob('*.json')))}; findings: {len(findings)}; roundings: {len(roundings)}; open: {len(open_items)}")
    return 0


def main(argv: list[str]) -> int:
    try:
        return build()
    except BuildError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
