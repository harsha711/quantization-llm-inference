#!/usr/bin/env bash
# Local helper: emits one E059 batch script with the committed source tree embedded (as E050/E057).
#   bash scripts/e059/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=c8
#   bash scripts/e059/submit.sh full PREREQ_DIR_NAME "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=c8 --dependency=afterok:JOB
# Retrieval: the job tars its BeeGFS directory to ARC home (e059-<dir>.tgz + .sha256), copied with scp.
set -euo pipefail
mode=${1:?prereq or full}; pre=${2:-}; note=${3:-}; [[ $mode == prereq || ( $mode == full && -n $pre && -n $note ) ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
paths=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e004 scripts/e005 scripts/e008
       scripts/e010 scripts/e047 scripts/e050 scripts/e059 results/e050_20261118/prereq-c8-343302/sass-tc-lib.txt
       results/e010_c4_20261107/verified-inputs.sha256)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e059-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$([[ $mode == prereq ]] && echo 02:00:00 || echo 04:00:00)
#SBATCH --output=/tmp/e059-$mode-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
COMMIT=$(cd "$root" && git rev-parse HEAD)
TARSHA=$(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | sha256sum | cut -d' ' -f1)
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e059-20261120
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";tar -C "$OUT" -czf "$HOME/e059-$(basename "$S").tgz" "$(basename "$S")";sha256sum "$HOME/e059-$(basename "$S").tgz" > "$HOME/e059-$(basename "$S").tgz.sha256";exit "$rc"' EXIT
echo "commit $COMMIT"
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
PRE=
if [[ -n $PRE_NAME ]]; then PRE="$OUT/$PRE_NAME"; [[ $(cat "$PRE/driver.exit") == 0 ]]; [[ -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"; fi
SRC=/tmp/e059-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | base64)
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
prun bash "$SRC/scripts/e059/inner.sh" "$MODE" "$SRC" "$S" "$PRE" > "$S/inner.log" 2>&1
(cd "$S" && find . -type f ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF2
