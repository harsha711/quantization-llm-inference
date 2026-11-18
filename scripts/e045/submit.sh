#!/usr/bin/env bash
# Local helper: emits the E045 batch script with the source tree (git HEAD) embedded.
#   scripts/e045/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=c7
#   scripts/e045/submit.sh full PREREQ_DIR_NAME "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=c7 --dependency=afterok:JOB
set -euo pipefail
mode=${1:?prereq or full}; pre=${2:-}; note=${3:-}; [[ $mode == prereq || ( $mode == full && -n $pre && -n $note ) ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
paths=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e004 scripts/e005 scripts/e008 scripts/e010 scripts/e045 results/e010_c4_20261107/verified-inputs.sha256)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e045-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$([[ $mode == prereq ]] && echo 02:00:00 || echo 01:30:00)
#SBATCH --output=/tmp/e045-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
TARSHA=$(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | sha256sum | cut -d' ' -f1)
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e045-20261117
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
SRC=/tmp/e045-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | base64)
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
PRE=; [[ -n $PRE_NAME ]] && PRE="$OUT/$PRE_NAME"
# Local inspection of the retrieved prerequisite is recorded here; inner.sh requires the marker.
[[ -z $PRE || -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"
prun bash "$SRC/scripts/e045/inner.sh" "$MODE" "$SRC" "$S" "$PRE" > "$S/inner.log" 2>&1
(cd "$S" && sha256sum $(ls | grep -v -E '^(FINISHED|RETRIEVED|driver\.(log|exit))$') > all.sha256)
EOF2
