# Logs

Build and test logs for the steps in this repo's history, copied from the machine that ran them. The names follow the
step names used in the commit messages and `gates/`:

| Prefix | Step |
|---|---|
| `build-89.w1`, `ctest-*.w1` | wave 1 |
| `build-89.w2`, `ctest-*.w2a` | wave 2a |
| `build-89.2b1` … `build-89.2b13` | wave 2b, step by step |
| `build-rename*` | the Halocline rename |
| `build-89.sm89`, `ctest-*.sm89` | the sm_89-only scope commit |
| `serve-tests.*`, `pytest-serve.*` | the Python server suite |

`ctest` logs record every test by name, its result and its time; the trailing summary lists the skips. A skipped test
names the reason in its own output (a card that is not sm_120, a pack that is absent, a two-GPU test on one card).

The build logs were taken with `ninja`, so they list one line per target. `build-rename.log` is the full 728-target
build; the later logs are incremental and say how many targets were rebuilt.
