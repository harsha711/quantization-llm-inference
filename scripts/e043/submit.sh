#!/usr/bin/env bash
# Local helper: emits the E043 batch script with the source tree embedded.  scripts/e043/submit.sh prereq|full | ssh arc-off-network sbatch --parsable
set -euo pipefail
mode=${1:?prereq or full}; [[ $mode == prereq || $mode == full ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e043-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$([[ $mode == prereq ]] && echo 00:30:00 || echo 01:00:00)
#SBATCH --output=/tmp/e043-%j.log
MODE=$mode
TARSHA=$(cd "$root" && git archive HEAD CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e043 | gzip -n | sha256sum | cut -d' ' -f1)
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e043-20261117
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
SRC=/tmp/e043-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
(cd "$root" && git archive HEAD CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e043 | gzip -n | base64)
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
prun bash "$SRC/scripts/e043/inner.sh" "$MODE" "$SRC" "$S" > "$S/inner.log" 2>&1
(cd "$S" && sha256sum $(ls | grep -v -E '^(FINISHED|RETRIEVED|driver\.(log|exit))$') > all.sha256)
EOF2
