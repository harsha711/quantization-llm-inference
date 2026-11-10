#!/usr/bin/env bash
# Local helper: keep a running allocation and start further generated job scripts inside it (nodes are scarce).
#   inalloc.sh NODE JOB hold                      pause the batch shell so the allocation stays until its time limit
#   inalloc.sh NODE JOB run TAG <submit args...>  run "$SUBMIT <submit args>" (default scripts/e028/submit.sh) inside it;
#                                                 its directory gets the suffix -TAG
#                                                 AFTER=DIR delays the start until DIR/FINISHED exists (a queue inside the allocation)
#   inalloc.sh NODE JOB release                   resume the batch shell; the job then ends on its own
set -euo pipefail
node=$1 job=$2 act=$3; shift 3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
remote() { ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes $node bash -s"; }
pid="pid=\$(pgrep -u \"\$USER\" -f \"job$job/slurm_script\" | head -1); [[ -n \$pid ]]"
case $act in
hold) remote <<<"$pid; kill -STOP \$pid; grep State /proc/\$pid/status" ;;
release) remote <<<"$pid; kill -CONT \$pid" ;;
run) tag=$1; shift
   "${SUBMIT:-$here/submit.sh}" "$@" | ssh -o BatchMode=yes arc-off-network "ssh -o BatchMode=yes $node 'cat > /tmp/inalloc-$job-$tag.sh'"
   remote <<<"$pid
while IFS= read -r -d '' kv; do export \"\$kv\" 2>/dev/null || true; done < /proc/\$pid/environ
[[ \$SLURM_JOB_ID == $job ]]; export E_TAG=-$tag
nohup setsid bash -c 'while [[ -n \$0 && ! -e \$0/FINISHED ]]; do sleep 10; done; exec bash \$1' '${AFTER:-}' /tmp/inalloc-$job-$tag.sh > /tmp/inalloc-$job-$tag.out 2>&1 < /dev/null &
echo started $tag in job $job" ;;
*) exit 2 ;;
esac
