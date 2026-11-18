#!/usr/bin/env bash
# The checks that need no GPU, in one command: what a reviewer can run on a clean checkout to verify a change before
# the GPU gates (docs/testing.md). Halocline has no hosted CI; this is the same list of checks, run locally.
#
#   tools/verify.sh [BUILD_DIR]
#
# Always: the four structure checks, the Python server suite, the tool tests, the converter and migration tests.
# With BUILD_DIR (an already built tree, e.g. build-w): every registered test carries a cpu or gpu label, and the
# cpu-labelled ctest suite. PYTHON picks the interpreter (default python3; it needs requirements-test.txt). Exit 0 only
# when every step passed; each step prints one PASS or FAIL line, and the summary repeats the failures.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
B=${1:-}
fails=()
step() {   # name command...
  local name=$1; shift
  local log; log=$(mktemp)
  if "$@" > "$log" 2>&1; then echo "PASS  $name"; else echo "FAIL  $name"; tail -n 20 "$log" | sed 's/^/      /'; fails+=("$name"); fi
  rm -f "$log"
}
step "platform boundary" tools/check_platform_boundary.sh
step "model layering" tools/check_model_layering.sh
step "engine log lines (stderr)" tools/check_stderr_logging.sh
step "build structure (.cu includes)" tools/check_cu_includes.sh
step "server suite" "$PY" -m unittest discover -s serve/tests -t . -q
step "tool tests" "$PY" -m unittest tools.test_bench_flags tools.test_config_examples tools.test_bench_fixtures tools.test_serve_launcher \
    tools.test_tokenizer tools.test_greedy_identity tools.test_package_release -q
step "converter tests" "$PY" -m unittest discover -s tools/convert -p 'test_*.py' -q
step "migration tests" "$PY" -m unittest discover -s tools/migrations -p 'test_*.py' -q
if [[ -n "$B" ]]; then
  # the conversation cache keeps 4 GiB of its volume free (src/core/conversation_store.cpp kSpaceFloor), so its test
  # cannot write on a small tmpfs /tmp; the engine's temp_dir() honours TMPDIR (src/platform/file.cpp) for this case
  if [[ -z "${TMPDIR:-}" ]] && (( $(df -Pk /tmp | awk 'NR==2 {print $4}') < 5 * 1024 * 1024 )); then
    export TMPDIR="$(cd "$B" && pwd)/verify-tmp"; mkdir -p "$TMPDIR"
    echo "NOTE  /tmp has under 5 GiB free: TMPDIR=$TMPDIR for the cpu suite"
  fi
  # the converter tests run under the Python CMake found at configure time; it needs requirements.txt
  cpy=$(sed -n 's/^_\{0,1\}Python3_EXECUTABLE:[A-Z]*=//p' "$B/CMakeCache.txt" | head -n 1)
  if [[ -n "$cpy" ]] && ! "$cpy" -c 'import regex, numpy' 2>/dev/null; then
    echo "NOTE  $B was configured with $cpy, which lacks requirements.txt (regex, numpy): convert_tests will fail."
    echo "      Reconfigure with -DPython3_EXECUTABLE=<a python with requirements.txt installed>."
  fi
  labels() {
    export LC_ALL=C
    local names='s/^  Test *#[0-9]*: //p'
    comm -23 <(ctest --test-dir "$B" -N | sed -n "$names" | sort -u) \
             <( { ctest --test-dir "$B" -N -L cpu; ctest --test-dir "$B" -N -L gpu; } | sed -n "$names" | sort -u) > /tmp/.halo-unlabelled.$$
    local n; n=$(wc -l < /tmp/.halo-unlabelled.$$)
    [[ $n -eq 0 ]] || { echo "tests with no cpu or gpu label (register them with halo_cpu_test or halo_gpu_test):"; cat /tmp/.halo-unlabelled.$$; }
    rm -f /tmp/.halo-unlabelled.$$; [[ $n -eq 0 ]]
  }
  step "every test carries a cpu or gpu label" labels
  step "cpu suite (ctest -L cpu)" ctest --test-dir "$B" -L cpu -j "$(nproc)" --output-on-failure
else
  echo "SKIP  label check and cpu suite (no BUILD_DIR given)"
fi
if (( ${#fails[@]} )); then echo "verify: ${#fails[@]} failed: ${fails[*]}"; exit 1; fi
echo "verify: all passed"
