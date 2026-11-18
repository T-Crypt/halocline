#!/usr/bin/env bash
# Halocline sparse non-regression gate: dense-side work (the rename, the KV format enum, the planner, the drafter
# levers, the q4_0 code moved out of the sparse kernels) must not cost the sparse path any speed or quality.
#
# REF = the served release ~/halocline/releases/d314a21a (bin/basalt, its own .venv).
# NEW = ~/halocline/releases/b22ecbb2 (d8-integration b22ecbb2, git archive; bin/halocline). Override with NEW=.
# Both read /data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt and are scored against
# /data/models/halocline/kld-bases/flashnext-bf16-base.kld.
#
# Usage: run.sh [OUTDIR]      (default: run-<local timestamp> beside this script; OUTDIR must be empty)
#
# RUNTIME ~95 min with the card to itself, from the published runs:
#   gate 1   6 min   2 engine loads at 152 s (halocline-kv-kld/int8/bench.err:21) + 900 tokens at ~104 tok/s with
#                    --spec 0 (halocline-sparse-serve/README.md:16)
#   gate 2  46 min   4 x 11.5 min: the published int8 kv-sweep run took 07:05:58-07:17:29 (halocline-kv-kld/sweep.log)
#   gate 3  28 min   2 loads + 3 x 768 requests: 64 tokens at 131.8 tok/s is 0.49 s (halocline-kv-kld/README.md:37)
#   gate 4  10 min   2 loads + 24 requests a build; 12 free-text replies of up to 1024 tokens are 7.8 s each at
#                    131.8 tok/s
#   setup, provenance, power reads, summary  5 min
#
# The caller holds a `berth train` lease: this script runs no berth command (nested leases are refused) and starts
# one engine process at a time. The sparse engine page-locks ~41 GiB (halocline-greedy-identity-final/README.md:18).
set -u

REF=$HOME/halocline/releases/d314a21a
NEW=${NEW:-$HOME/halocline/releases/b22ecbb2}   # 2026-11-17 claude: the release tree that ships (W10 wrote the worktree; same commit b22ecbb2)
PACK=/data/models/halocline/packs/QFN-UD-IQ4_XS-v3.1-compat-mtp.basalt
BASE=/data/models/halocline/kld-bases/flashnext-bf16-base.kld
PAIRED=$HOME/homelab/state/evals/2026-11-17/halocline-dense-quality/kv_paired.py
PY=$NEW/.venv/bin/python
ALIAS=qfn-halocline
KV=int8
SPEC=8
MIN_AVAIL_GB=50
NO_THINK='{"chat_template_kwargs":{"enable_thinking":false}}'

case ${1:-} in
    /*) OUT=$1 ;;
    "") OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-$(date +%Y%m%dT%H%M%S) ;;
    *) OUT=$PWD/$1 ;;
esac
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "refusing: $OUT is not empty (give a new OUTDIR)" >&2
    exit 3
fi
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 3; }
LOG=$OUT/run.log
: > "$LOG" || { echo "cannot write $LOG" >&2; exit 3; }

SRV_PID=""
SRV_BIN=""
URL=""

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

show() {                                    # the command, quoted, exactly as run
    local s="" w
    for w in "$@"; do s+=$(printf '%q ' "$w"); done
    say "cmd: $s"
}

run_to() {                                  # run_to OUT ERR CMD...: stdout, stderr and the rc, all logged
    local out=$1 err=$2 rc
    shift 2
    show "$@"
    "$@" >"$out" 2>"$err"
    rc=$?
    say "rc=$rc -> $out"
    return "$rc"
}

probe() {                                   # probe NAME CMD...: the command, its output and its rc in the log
    local name=$1 out rc
    shift
    show "$@"
    out=$("$@" 2>&1)
    rc=$?
    printf '%s\n' "$out" | sed 's/^/  /' | tee -a "$LOG"
    say "rc=$rc -> $name"
    return "$rc"
}

free_gb() { free -g | awk '$1 == "Mem:" { print $7 }'; }

cleanup() {
    local i=0
    if [ -n "${SRV_PID:-}" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        say "trap: stopping server pid $SRV_PID"
        kill -TERM "$SRV_PID" 2>/dev/null || true
        while kill -0 "$SRV_PID" 2>/dev/null && [ "$i" -lt 60 ]; do sleep 1; i=$((i + 1)); done
        kill -KILL "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Everything this gate starts, matched by argv, so a stray engine cannot share the card. Written so pgrep cannot match
# its own command line.
IDLE_PATTERN='[e]ngine strata|[b]asalt --serve|[h]alocline --serve|[b]asalt serve |[h]alocline serve |[b]asalt-bench|[h]alocline-bench'
assert_idle() {
    local found
    found=$(pgrep -af "$IDLE_PATTERN" 2>/dev/null || true)
    if [ -n "$found" ]; then
        say "refusing: another engine or bench is running:"
        printf '%s\n' "$found" | sed 's/^/  /' | tee -a "$LOG"
        return 1
    fi
    return 0
}

free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

SERVER_FLAGS=(--host 127.0.0.1 --alias "$ALIAS" --kv "$KV" --spec "$SPEC" --prefill auto
              --expert-cache 7000 --adapt-swaps 0 --max-context 8192)

start_server() {                            # start_server ref|new PORT LOG
    local which=$1 port=$2 log=$3 bin py
    local -a cmd
    case $which in
        ref) bin=$REF/bin/basalt; py=$REF/.venv/bin/python ;;
        new) bin=$NEW/bin/halocline; py=$NEW/.venv/bin/python ;;
        *) say "start_server: unknown build $which"; return 1 ;;
    esac
    cmd=("$bin" serve "$PACK" --python "$py" "${SERVER_FLAGS[@]}" --port "$port")
    show "${cmd[@]}"
    "${cmd[@]}" >>"$log" 2>&1 &
    SRV_PID=$!
    SRV_BIN=$bin
    URL=http://127.0.0.1:$port
    say "$which server: pid $SRV_PID at $URL (log $log)"
}

wait_loaded() {                             # wait_loaded PORT [TIMEOUT]: /health loaded=true, or the pid dies
    local port=$1 timeout=${2:-900} t0=$SECONDS last="" h loaded loading pct
    while [ $((SECONDS - t0)) -lt "$timeout" ]; do
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            say "health: server pid $SRV_PID exited while loading"
            return 1
        fi
        h=$(curl -fsS --max-time 5 "http://127.0.0.1:$port/health" 2>/dev/null) || h=""
        if [ -n "$h" ]; then
            loaded=$(printf '%s' "$h" | jq -r '.loaded // false')
            loading=$(printf '%s' "$h" | jq -r '.loading // false')
            pct=$(printf '%s' "$h" | jq -r '.percent // "-"')
            if [ "$loaded" = "true" ]; then
                say "health: loaded=true after $((SECONDS - t0)) s"
                return 0
            fi
            if [ "$pct/$loading" != "$last" ]; then
                say "health: loading=$loading percent=$pct"
                last=$pct/$loading
            fi
        fi
        sleep 5
    done
    say "health: not loaded after ${timeout} s"
    return 1
}

stop_server() {
    local pid=${SRV_PID:-} i=0
    [ -n "$pid" ] || return 0
    say "stopping server pid $pid"
    kill -TERM "$pid" 2>/dev/null || true
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 120 ]; do sleep 1; i=$((i + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        say "server pid $pid ignored SIGTERM for 120 s: SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 2
    fi
    wait "$pid" 2>/dev/null || true
    SRV_PID=""
    # the engine is the server's child; if it outlived the server, stop it here and say so
    i=0
    while pgrep -f "^$SRV_BIN --serve" >/dev/null 2>&1 && [ "$i" -lt 30 ]; do
        if [ $i -eq 0 ]; then
            say "engine of $SRV_BIN outlived the server: SIGTERM"
            pkill -TERM -f "^$SRV_BIN --serve" 2>/dev/null || true
        fi
        sleep 1
        i=$((i + 1))
    done
    if pgrep -f "^$SRV_BIN --serve" >/dev/null 2>&1; then
        say "engine still running after 30 s: SIGKILL"
        pkill -KILL -f "^$SRV_BIN --serve" 2>/dev/null || true
        sleep 2
    fi
    say "server stopped; $(free_gb) GiB available"
}

# ---- gate 1: greedy identity, the published protocol (halocline-greedy-identity-final/README.md:11-17)
gate1() {
    local d=$OUT/1-greedy t0=$SECONDS rc
    mkdir -p "$d/tmp"
    say "gate 1: greedy identity, 900 tokens, spec 0, expert cache 7000, prefill auto"
    run_to "$d/greedy.txt" "$d/greedy.err" env TMPDIR="$d/tmp" "$PY" "$NEW/tools/greedy_identity.py" \
        "$REF/bin/basalt" "$NEW/bin/halocline" --model "$PACK" -- --spec 0 --expert-cache 7000 --prefill auto
    rc=$?
    printf '%s\n' "$rc" > "$d/rc"
    say "gate 1: $rc, done in $((SECONDS - t0)) s"
    tail -n 3 "$d/greedy.txt" | sed 's/^/  /' | tee -a "$LOG"
}

# ---- gate 2: KLD and speed, the kv-sweep command verbatim (halocline-kv-kld/kv-sweep.sh:14-17), ABBA
gate2() {
    local d=$OUT/2-kv t0=$SECONDS t1 pair p bin
    mkdir -p "$d"
    for pair in ref1 new1 ref2 new2; do
        bin=$NEW/bin/halocline-bench
        case $pair in ref*) bin=$REF/bin/basalt-bench ;; esac
        mkdir -p "$d/$pair"
        assert_idle || return 1
        t1=$SECONDS
        say "gate 2: kv-sweep run $pair"
        run_to "$d/$pair/bench.json" "$d/$pair/bench.err" "$bin" --model "$PACK" -t conv,struct,prose \
            -p 32768,131072 --seed 1 --quality 100 --kld-base "$BASE" --logs "$d/$pair" -o json -- \
            --kv "$KV" --spec "$SPEC" --prefill auto
        say "gate 2: $pair done in $((SECONDS - t1)) s"
    done
    for p in ref1-new1 ref2-new2 ref1-ref2 new1-new2; do
        run_to "$d/paired-$p.md" "$d/paired-$p.err" "$PY" "$PAIRED" "$d" "${p%%-*}" "${p#*-}"
    done
    say "gate 2: done in $((SECONDS - t0)) s"
    sed 's/^/  /' "$d/paired-ref1-new1.md" | tee -a "$LOG"
}

run_sampling() {                            # run_sampling OUT SEEDS LABEL
    local out=$1 seeds=$2 label=$3
    say "gate 3: sampling run $label (seeds $seeds)"
    run_to "${out%.jsonl}.log" "${out%.jsonl}.err" "$PY" "$NEW/tools/sampling_dist/sampling_dist.py" run \
        --url "$URL" --model "$ALIAS" --prompts "$NEW/tools/sampling_dist/prompts.jsonl" \
        --seeds "$seeds" --out "$out" --label "$label" --extra "$NO_THINK"
}

# ---- gate 3: sampling distribution, P0.2 (tools/sampling_dist/sampling_dist.py:1-17; docs/testing.md:32)
gate3() {
    local d=$OUT/3-sampling t0=$SECONDS port rc
    mkdir -p "$d"
    port=$(free_port)
    assert_idle || return 1
    start_server ref "$port" "$d/ref-server.log" || return 1
    if ! wait_loaded "$port"; then stop_server; return 1; fi
    run_sampling "$d/ref-a.jsonl" 0-31 ref-a
    run_sampling "$d/ref-b.jsonl" 32-63 ref-b
    stop_server || return 1
    port=$(free_port)
    assert_idle || return 1
    start_server new "$port" "$d/new-server.log" || return 1
    if ! wait_loaded "$port"; then stop_server; return 1; fi
    run_sampling "$d/new.jsonl" 0-31 new
    stop_server || return 1
    say "gate 3: compare ref-a ref-b new"
    run_to "$d/compare.txt" "$d/compare.err" "$PY" "$NEW/tools/sampling_dist/sampling_dist.py" compare \
        "$d/ref-a.jsonl" "$d/ref-b.jsonl" "$d/new.jsonl"
    rc=$?
    printf '%s\n' "$rc" > "$d/compare.rc"
    say "gate 3: done in $((SECONDS - t0)) s"
    sed 's/^/  /' "$d/compare.txt" | tee -a "$LOG"
}

# ---- gate 4: tool-call breadth, P0.4 (tools/tool_calls/tool_calls.py:1-13; docs/testing.md:34)
gate4() {
    local d=$OUT/4-tools t0=$SECONDS port
    mkdir -p "$d"
    port=$(free_port)
    assert_idle || return 1
    start_server ref "$port" "$d/ref-server.log" || return 1
    if ! wait_loaded "$port"; then stop_server; return 1; fi
    say "gate 4: REF tool calls"
    run_to "$d/ref.log" "$d/ref.err" "$PY" "$NEW/tools/tool_calls/tool_calls.py" run \
        --url "$URL" --model "$ALIAS" --out "$d/ref.jsonl" --label ref --extra "$NO_THINK"
    stop_server || return 1
    port=$(free_port)
    assert_idle || return 1
    start_server new "$port" "$d/new-server.log" || return 1
    if ! wait_loaded "$port"; then stop_server; return 1; fi
    say "gate 4: NEW tool calls"
    run_to "$d/new.log" "$d/new.err" "$PY" "$NEW/tools/tool_calls/tool_calls.py" run \
        --url "$URL" --model "$ALIAS" --out "$d/new.jsonl" --label new --extra "$NO_THINK"
    stop_server || return 1
    say "gate 4: score"
    run_to "$d/score-ref.txt" "$d/score-ref.err" "$PY" "$NEW/tools/tool_calls/tool_calls.py" score "$d/ref.jsonl"
    run_to "$d/score-new.txt" "$d/score-new.err" "$PY" "$NEW/tools/tool_calls/tool_calls.py" score \
        "$d/new.jsonl" --ref "$d/ref.jsonl"
    say "gate 4: done in $((SECONDS - t0)) s"
}

# ---------------------------------------------------------------------------------------------------------------
say "halocline sparse non-regression gate: REF=$REF NEW=$NEW"
say "pack=$PACK"
say "kld base=$BASE"
for c in awk curl jq pgrep pkill python3 sha256sum; do
    command -v "$c" >/dev/null 2>&1 || { echo "refusing: $c not found" | tee -a "$LOG" >&2; exit 3; }
done
probe "NEW commit" git -C "$NEW" rev-parse HEAD
probe "NEW status (empty = clean)" git -C "$NEW" status --porcelain
probe "REF binary" sha256sum "$REF/bin/basalt"
probe "NEW binary" sha256sum "$NEW/bin/halocline"
probe "free -g (start)" free -g
if command -v nvidia-smi >/dev/null 2>&1; then
    probe "power limit (start)" nvidia-smi --query-gpu=power.limit,power.max_limit,driver_version --format=csv,noheader
else
    say "nvidia-smi not found; the bench records the power limit in its JSON"
fi

for f in "$PACK" "$BASE" "$REF/bin/basalt" "$REF/bin/basalt-bench" "$REF/.venv/bin/python" \
         "$NEW/bin/halocline" "$NEW/bin/halocline-bench" "$NEW/.venv/bin/python" \
         "$NEW/tools/greedy_identity.py" "$NEW/tools/sampling_dist/sampling_dist.py" \
         "$NEW/tools/tool_calls/tool_calls.py" "$PAIRED"; do
    [ -e "$f" ] || { say "refusing: missing $f"; exit 3; }
done
for f in conv struct prose prefill; do
    a=$(sha256sum "$REF/data/bench/$f.ids" 2>/dev/null | cut -d' ' -f1)
    b=$(sha256sum "$NEW/data/bench/$f.ids" 2>/dev/null | cut -d' ' -f1)
    [ -n "$a" ] && [ "$a" = "$b" ] || { say "refusing: data/bench/$f.ids differs between REF and NEW"; exit 3; }
done
say "inputs: pack, KLD base, binaries, tools and the four bench fixtures are present; fixtures hash-identical"

avail=$(free_gb)
case $avail in
    ''|*[!0-9]*) say "cannot parse free -g output"; exit 3 ;;
esac
if [ "$avail" -lt "$MIN_AVAIL_GB" ]; then
    say "refusing: ${avail} GiB available, the sparse engine page-locks ~41 GiB and needs ${MIN_AVAIL_GB} GiB free"
    exit 3
fi
say "memory: ${avail} GiB available (>= ${MIN_AVAIL_GB})"
assert_idle || exit 3

gate1
gate2
gate3
gate4

TOTAL=$SECONDS
say "all gates finished in ${TOTAL} s; summary: $OUT/summary.md"
if command -v nvidia-smi >/dev/null 2>&1; then
    probe "power limit (end)" nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm --format=csv,noheader
fi
probe "free -g (end)" free -g

# --- summary generator (python3) ---
python3 - "$OUT" "$REF" "$NEW" "$(date -Is)" "$TOTAL" "$PACK" <<'PY' >"$OUT/summary.console" 2>&1
import json
import re
import sys
from pathlib import Path

OUT, REF, NEW, STAMP, TOTAL, PACK = (Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4],
                                     sys.argv[5], sys.argv[6])
TOL = 3.0        # percent; the protocol's own run-to-run stddev is 0.8-2.1% (halocline-kv-kld/spec8/bench.json)
PUB = [("conv tok/s", "tg conv", 131.8), ("struct tok/s", "tg struct", 264.6), ("prose tok/s", "tg prose", 133.7),
       ("prefill 32K t/s", "pp32768", 6053.0), ("prefill 131K t/s", "pp131072", 6019.0)]
PUB_KLD, PUB_TOP1 = 0.08195, 90.45


def text(path):
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return ""


def bench(path):
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return {"speed": {r.get("test"): r.get("tok_s_mean") for r in d.get("summary") or []},
            "q": d.get("quality") or {}, "power": d.get("power_limit")}


runs = {n: bench(OUT / "2-kv" / n / "bench.json") for n in ("ref1", "ref2", "new1", "new2")}


def vals(names, section, key):
    out = [runs[n][section].get(key) for n in names if runs.get(n)]
    return [v for v in out if isinstance(v, (int, float))]


def avg(names, section, key):
    v = vals(names, section, key)
    return sum(v) / len(v) if v else None


def ci(path, label):
    """The paired 95% interval of a kv_paired.py table, for the row named `label`."""
    for line in text(path).splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 8 and c[0] == label:
            m = re.search(r"\[\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", c[6])
            if m:
                return float(m.group(1)), float(m.group(2))
    return None


def f(x, spec="{:.1f}"):
    return "n/a" if x is None else spec.format(x)


g1_text = text(OUT / "1-greedy" / "greedy.txt")
g1_rc = text(OUT / "1-greedy" / "rc").strip()
tokens = re.findall(r"(?:ref|new): (\d+) tokens", g1_text)
g1_ok = g1_rc == "0" and "GREEDY IDENTICAL" in g1_text and len(tokens) == 2 and tokens[0] == tokens[1] == "900"
g1_num = (("%s/%s tokens, " % (tokens[0], tokens[1])) +
          ("GREEDY IDENTICAL" if "GREEDY IDENTICAL" in g1_text else "not identical")) \
    if len(tokens) == 2 else "no tool output"

ci1 = ci(OUT / "2-kv" / "paired-ref1-new1.md", "new1")
ci2 = ci(OUT / "2-kv" / "paired-ref2-new2.md", "new2")
speed = []
for label, key, pub in PUB:
    r1 = runs["ref1"]["speed"].get(key) if runs.get("ref1") else None
    r2 = runs["ref2"]["speed"].get(key) if runs.get("ref2") else None
    n1 = runs["new1"]["speed"].get(key) if runs.get("new1") else None
    n2 = runs["new2"]["speed"].get(key) if runs.get("new2") else None
    a, b = avg(("ref1", "ref2"), "speed", key), avg(("new1", "new2"), "speed", key)
    speed.append((label, r1, r2, a, n1, n2, b, (100.0 * (b - a) / a) if a and b else None, pub))
speed_ok = len(speed) == len(PUB) and all(s[7] is not None and abs(s[7]) <= TOL for s in speed)
kld_ok = all(c is not None and c[0] <= 0.0 <= c[1] for c in (ci1, ci2))
kld_ref, kld_new = avg(("ref1", "ref2"), "q", "kld_mean"), avg(("new1", "new2"), "q", "kld_mean")
worst = max((abs(s[7]) for s in speed if s[7] is not None), default=None)
g2_ok = bool(kld_ok and speed_ok)
g2_num = "KLD REF %s / NEW %s (paired %s); worst speed delta %s%% (limit %.0f%%)" % (
    f(kld_ref, "{:.5f}"), f(kld_new, "{:.5f}"),
    ("[%+.5f, %+.5f]" % ci1) if ci1 else "n/a", f(worst), TOL)

cmp_lines = [x.strip() for x in text(OUT / "3-sampling" / "compare.txt").splitlines() if x.strip()]
cmp_rc = text(OUT / "3-sampling" / "compare.rc").strip()
g3_ok = cmp_rc == "0" and bool(cmp_lines) and cmp_lines[-1] == "PASS"
metric_verdicts = []
for line in cmp_lines:
    m = re.match(r"^(top1|top10|logprob|first_tv)\s+.*\s+(pass|FAIL)$", line)
    if m:
        metric_verdicts.append("%s %s" % (m.group(1), m.group(2).lower()))
g3_num = ", ".join(metric_verdicts) if metric_verdicts else (cmp_lines[-1] if cmp_lines else "no compare output")


def rights(path):
    out = {}
    for line in text(path).splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "id" in r and "right" in r:
            out[r["id"]] = bool(r["right"])
    return out


ref_r, new_r = rights(OUT / "4-tools" / "ref.jsonl"), rights(OUT / "4-tools" / "new.jsonl")
regressed = sorted(i for i, ok in ref_r.items() if ok and not new_r.get(i, False))
g4_ok = bool(ref_r) and bool(new_r) and set(ref_r) == set(new_r) and not regressed
g4_num = "REF %d/%d, NEW %d/%d, regressed %s" % (
    sum(ref_r.values()), len(ref_r), sum(new_r.values()), len(new_r), ", ".join(regressed) or "none")

rows = [("1", "greedy identity (spec 0)", g1_ok, g1_num),
        ("2", "KLD + speed (int8 KV, spec 8)", g2_ok, g2_num),
        ("3", "sampling distribution (temp 1.0)", g3_ok, g3_num),
        ("4", "tool-call breadth (temp 0)", g4_ok, g4_num)]
ok = all(r[2] for r in rows)
power = sorted({r["power"] for r in runs.values() if r and r.get("power")})
L = ["# Halocline sparse non-regression gate, %s" % STAMP, "",
     "REF `%s` (bin/basalt) vs NEW `%s` (bin/halocline). Pack `%s`." % (REF, NEW, PACK),
     "Quality base `flashnext-bf16-base.kld`; int8 KV; drafter `--spec 8` except gate 1 (`--spec 0`).",
     "Power limit %s. Wall time %s s." % (", ".join(power) or "not recorded", TOTAL), "",
     "| # | gate | verdict | numbers |", "|---|---|---|---|"]
L += ["| %s | %s | %s | %s |" % (r[0], r[1], "PASS" if r[2] else "FAIL", r[3]) for r in rows]
L += ["", "## 1. Greedy identity detail", "", "```", g1_text.strip() or "(missing)", "```", ""]
L += ["## 2. KLD and speed detail", "",
      "The published int8 numbers for comparison: KLD %s, top-1 %s%%, conv 131.8 / struct 264.6 / prose" % (
          f(PUB_KLD, "{:.5f}"), f(PUB_TOP1, "{:.2f}")),
      "133.7 tok/s, prefill 6,053 / 6,019 t/s (halocline-kv-kld/README.md:25,37).", "",
      "| metric | REF1 | REF2 | REF mean | NEW1 | NEW2 | NEW mean | NEW-REF % | published int8 |",
      "|---|---|---|---|---|---|---|---|---|"]
L += ["| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (s[0], f(s[1]), f(s[2]), f(s[3]), f(s[4]), f(s[5]),
                                                          f(s[6]), ("%+.1f" % s[7]) if s[7] is not None else "n/a",
                                                          f(s[8])) for s in speed]
L += ["", "| run | chunks | rows | PPL | mean KLD | top-1 % |", "|---|---|---|---|---|---|"]
for n in ("ref1", "ref2", "new1", "new2"):
    q = runs[n]["q"] if runs.get(n) else {}
    L.append("| %s | %s | %s | %s | %s | %s |" % (n, q.get("chunks", "n/a"), q.get("rows", "n/a"),
                                                  f(q.get("ppl"), "{:.4f}"), f(q.get("kld_mean"), "{:.5f}"),
                                                  f(q.get("top1_pct"), "{:.2f}")))
L += ["", "Paired KLD (kv_paired.py): the 95% interval of NEW minus REF, resampling chunks. Gate 2 needs it to hold",
      "0 on both pairs, and every speed metric inside %.0f%% on the ABBA means." % TOL, ""]
for p in ("ref1-new1", "ref2-new2", "ref1-ref2", "new1-new2"):
    L += ["`paired-%s.md`:" % p, "", text(OUT / "2-kv" / ("paired-%s.md" % p)).strip() or "(missing)", ""]
L += ["## 3. Sampling detail", "", "```", text(OUT / "3-sampling" / "compare.txt").strip() or "(missing)", "```", ""]
L += ["## 4. Tool-call detail", "", "REF:", "", "```", text(OUT / "4-tools" / "score-ref.txt").strip() or "(missing)",
      "```", "", "NEW against REF:", "", "```", text(OUT / "4-tools" / "score-new.txt").strip() or "(missing)",
      "```", ""]
L += ["## Acceptance", "",
      "* Gate 1: `greedy_identity.py` exit 0, `GREEDY IDENTICAL`, 900 of 900 tokens on both sides.",
      "* Gate 2: the paired 95% interval of NEW minus REF holds 0 on both ABBA pairs, and every speed metric is",
      "  within %.0f%% on the ABBA means. A REF that misses the published numbers is shown in the table, not scored." % TOL,
      "* Gate 3: `sampling_dist.py compare` exits 0 and its last line is `PASS` (top1, top10, logprob and first_tv",
      "  inside the REF build's own seed-change interval).",
      "* Gate 4: no case REF answers right is wrong on NEW (the per-case JSONL comparison); accuracy is reported.",
      "* Gate 1 and 2 configurations are the published ones verbatim; gates 3 and 4 pin placement",
      "  (`--expert-cache 7000 --adapt-swaps 0 --max-context 8192`) so adaptation cannot move experts between runs.",
      ""]
(OUT / "summary.md").write_text("\n".join(L) + "\n")
print("| # | gate | verdict | numbers |")
print("|---|---|---|---|")
for r in rows:
    print("| %s | %s | %s | %s |" % (r[0], r[1], "PASS" if r[2] else "FAIL", r[3]))
print("wrote %s" % (OUT / "summary.md"))
sys.exit(0 if ok else 1)
PY

rc=$?
cat "$OUT/summary.console" | tee -a "$LOG"
if [ "$rc" -eq 0 ]; then
    say "gate: all four PASS"
else
    say "gate: FAIL, read $OUT/summary.md"
fi
exit "$rc"
