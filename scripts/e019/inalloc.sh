#!/usr/bin/env bash
# Local helper: run a later E019 stage INSIDE the already-running allocation JOB on c7 instead of
# a new sbatch job (the queue is FIFO and a released node goes to ~25 waiting jobs).
#   scripts/e019/inalloc.sh hold JOB             pause the finished batch shell so the allocation stays
#   scripts/e019/inalloc.sh validate|primary|heldout JOB
#   scripts/e019/inalloc.sh release JOB          resume the batch shell; job then exits with its own status
# The stage script is the unchanged submit.sh output, started with the batch shell's own environment.
set -euo pipefail
stage=$1 job=$2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
c7() { ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes c7 bash -s -- $job $stage"; }
pid='pid=$(pgrep -u "$USER" -f "job$1/slurm_script" | head -1); [[ -n $pid ]]'
case $stage in
hold) c7 <<<"$pid; kill -STOP \$pid; grep State /proc/\$pid/status" ;;
release) c7 <<<"$pid; touch /mnt/beegfs/hpuvvad/csc548-w4a16-e019-20261109/smoke-\$1/RETRIEVED; kill -CONT \$pid" ;;
*) "$here/submit.sh" "$stage" "$job" | ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes c7 'cat > /tmp/e019-$stage-$job.sh'"
   c7 <<<"$pid
while IFS= read -r -d '' kv; do export \"\$kv\"; done < /proc/\$pid/environ
[[ \$SLURM_JOB_ID == \$1 ]]
nohup setsid bash /tmp/e019-\$2-\$1.sh > /tmp/e019-\$2-\$1.out 2>&1 < /dev/null &
echo started \$2 in job \$1" ;;
esac
