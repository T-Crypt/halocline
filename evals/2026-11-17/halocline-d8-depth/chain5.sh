#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$HOME/.local/share/mise/shims:/usr/local/bin:/usr/bin:/bin
D=$(cd "$(dirname "$0")" && pwd)
while systemctl --user is-active -q halocline-d8-w4-chain; do sleep 30; done
echo "$(date -Is) window 4 done; taking window 5" >> "$D/run.log"; sleep 10
exec berth train --as halocline-d8-w5 -- "$D/window5.sh" > "$D/berth-w5.log" 2>&1
