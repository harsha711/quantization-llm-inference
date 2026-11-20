#!/usr/bin/env bash
# Local helper: emits one E057 batch script with the committed source tree embedded (as E041). MODE: prereq | full.
#   scripts/e057/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=c8
#   scripts/e057/submit.sh full PREREQ_DIR "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=c8 --dependency=afterok:JOB
# E057: measured CPU counters/clocks/STREAM (E041 cpu_bench, unchanged) and GPU bandwidth/tensor roofs (new microbenchmarks).
set -euo pipefail
mode=$1 pre=${2:-} note=${3:-}
[[ $mode == prereq || ( $mode == full && -n $pre && -n $note ) ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
files=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e057)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e057-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$([[ $mode == prereq ]] && echo 00:45:00 || echo 01:30:00)
#SBATCH --output=/tmp/e057-$mode-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
COMMIT=$(cd "$root" && git rev-parse HEAD)
TARSHA=$(cd "$root" && git archive HEAD "${files[@]}" | gzip -n | sha256sum | cut -d' ' -f1)
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e057-20261120
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
# Copy the finished directory to home for retrieval from the login node (as E056).
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";tar -C "$OUT" -czf "$HOME/e057-$(basename "$S").tgz" "$(basename "$S")";sha256sum "$HOME/e057-$(basename "$S").tgz" > "$HOME/e057-$(basename "$S").tgz.sha256";exit "$rc"' EXIT
echo "commit $COMMIT"
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
PRE=
if [[ -n $PRE_NAME ]]; then PRE="$OUT/$PRE_NAME"; [[ $(cat "$PRE/driver.exit") == 0 ]]; [[ -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"; fi
SRC=/tmp/e057-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
(cd "$root" && git archive HEAD "${files[@]}" | gzip -n | base64)
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
prun bash "$SRC/scripts/e057/inner.sh" "$MODE" "$SRC" "$S" "$PRE" > "$S/inner.log" 2>&1
(cd "$S" && find . -type f ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF2
