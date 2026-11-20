#!/usr/bin/env bash
# Local helper: emits the E058 batch script with the committed source tree embedded (as E050/E057).
# Stage E050's library first (unchanged bytes, hash-checked in the job):
#   ssh arc-off-network mkdir -p csc548-w4a16-e058-20261120
#   scp results/e050_20261118/prereq-c8-343302/libe050_tc.so arc-off-network:csc548-w4a16-e058-20261120/
#   bash scripts/e058/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=c8
#   bash scripts/e058/submit.sh full PREREQ_DIR_NAME "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=c8 --dependency=afterok:JOB
set -euo pipefail
mode=${1:?prereq or full}; pre=${2:-}; note=${3:-}; [[ $mode == prereq || ( $mode == full && -n $pre && -n $note ) ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
paths=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e004 scripts/e005 scripts/e008
       scripts/e010 scripts/e058 results/e010_c4_20261107/verified-inputs.sha256 results/e050_20261118/summary.json
       results/e050_20261118/prereq-c8-343302/binaries.sha256 results/e050_20261118/full-c8-343308/series2.jsonl
       scripts/checks/floorcheck.py)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e058-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/e058-$mode-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
COMMIT=$(cd "$root" && git rev-parse HEAD)
TARSHA=$(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | sha256sum | cut -d' ' -f1)
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e058-20261120
LIB=$HOME/csc548-w4a16-e058-20261120/libe050_tc.so
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
# Copy the finished directory to home for retrieval from the login node (as E056/E057).
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";tar -C "$OUT" -czf "$HOME/e058-$(basename "$S").tgz" "$(basename "$S")";sha256sum "$HOME/e058-$(basename "$S").tgz" > "$HOME/e058-$(basename "$S").tgz.sha256";exit "$rc"' EXIT
echo "commit $COMMIT"
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
PRE=
if [[ -n $PRE_NAME ]]; then PRE="$OUT/$PRE_NAME"; [[ $(cat "$PRE/driver.exit") == 0 ]]; [[ -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"; fi
SRC=/tmp/e058-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | base64)
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
prun bash "$SRC/scripts/e058/inner.sh" "$MODE" "$SRC" "$S" "$LIB" "$PRE" > "$S/inner.log" 2>&1
(cd "$S" && find . -type f ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF2
