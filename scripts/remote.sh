#!/usr/bin/env bash
# Sync this repo to ARC and run a command on a compute node, then pull results/ back.
#   scripts/remote.sh scripts/build.sh
#   NODE=c27 scripts/remote.sh ./build/w4a16_bench --M 16
# Requires a running allocation on $NODE (e.g. an interactive salloc job).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${ARC_KEY:-$HOME/username}"
LOGIN="${ARC_LOGIN:-hpuvvad@arc.csc.ncsu.edu}"
NODE="${NODE:-c7}"
DIR="${REMOTE_DIR:-csc548-w4a16}"
SSH=(ssh -i "$KEY" -o BatchMode=yes)

rsync -az --delete --exclude 'build*/' --exclude .git --exclude results -e "${SSH[*]}" \
  "$ROOT/" "$LOGIN:$DIR/"
cmd="cd ~/$DIR && export PATH=/usr/local/cuda/bin:\$PATH && mkdir -p results && $(printf '%q ' "$@")"
status=0
"${SSH[@]}" "$LOGIN" "ssh -o BatchMode=yes $NODE $(printf '%q' "$cmd")" || status=$?
rsync -az -e "${SSH[*]}" "$LOGIN:$DIR/results/" "$ROOT/results/"
exit $status
