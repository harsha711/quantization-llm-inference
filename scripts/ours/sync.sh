#!/usr/bin/env bash
# Move the roofline-series work between the Mac and ARC (uses the `arc` Host entry
# in ~/.ssh/config; run in your own terminal, since the key has a passphrase).
#
#   bash scripts/ours/sync.sh push   # source -> arc:~/w4a16-ours/ (no build/, results/, .git)
#   bash scripts/ours/sync.sh pull   # arc:~/w4a16-ours/results/ours_* -> results/
#
# No --delete: nothing on either side is ever removed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE="${REMOTE:-arc:w4a16-ours}"
case "${1:-}" in
  push)
    rsync -rtv --exclude 'build*/' --exclude results/ --exclude .git/ --exclude __pycache__/ \
      --exclude .DS_Store "$ROOT/" "$REMOTE/"
    ;;
  pull)
    mkdir -p "$ROOT/results"
    rsync -rtv --include 'ours_*/' --include 'ours_*/**' --exclude '*' \
      "$REMOTE/results/" "$ROOT/results/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 1
    ;;
esac
