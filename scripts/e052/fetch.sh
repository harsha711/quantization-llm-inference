#!/usr/bin/env bash
# Local helper: wait until an E052 stage directory on BeeGFS is FINISHED, copy it (small files; *.npz logit archives stay on
# BeeGFS, listed in its all.sha256) into results/e052_20261118/, then release the job.  fetch.sh NODE DIRNAME
set -euo pipefail
node=$1 dir=$2 D=/mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118
dest="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/results/e052_20261118"; mkdir -p "$dest"
until ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes $node test -e $D/$dir/FINISHED" 2>/dev/null; do sleep 60; done
ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes $node 'tar -C $D -cf - --exclude=*.npz $dir'" | tar -x -C "$dest"
ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes $node touch $D/$dir/RETRIEVED"
echo "fetched $dir exit $(cat "$dest/$dir/driver.exit")"
