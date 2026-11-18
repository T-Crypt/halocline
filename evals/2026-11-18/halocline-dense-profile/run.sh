#!/usr/bin/env bash
# W13 dense profiler on the GPU (Halocline branch dense-profile c10dd83a). See profile.py.
# 08:30 first try: bin/halocline-vision not built (failed-0830/). 08:30:45 second try: smoke --long through one
# server; it passed, but the engine's stderr goes to the server's log buffer and the tables were lost (try2-0830/).
# Run under: berth train --as halocline-dense-profile -- run.sh
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
O=$(cd "$(dirname "$0")" && pwd)
echo "$(date -Is) start $(git -C "$HOME/wt/halocline-dense-profile" rev-parse --short HEAD)" >> "$O/run.log"
python3 "$O/profile.py" "$O/out" >> "$O/run.log" 2>&1
echo "$(date -Is) end rc=$?" >> "$O/run.log"
