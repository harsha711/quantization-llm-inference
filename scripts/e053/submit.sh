#!/usr/bin/env bash
# Emit a self-contained batch script. Submit from ARC home with sbatch, never srun.
set -euo pipefail
mode=${1:?baseline|validation|full}; pre=${2:-}
[[ $mode == baseline || $mode == validation || ( $mode == full && -n $pre ) ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
commit=$(git -C "$root" rev-parse --short HEAD)
paths=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/ours/check.sh scripts/e004 scripts/e005 scripts/e008 scripts/e010 scripts/e045 scripts/e049 scripts/e053 results/e010_c4_20261107/verified-inputs.sha256)
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e053-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/tmp/e053-%j.log
MODE=$mode PRE_NAME=$pre
export PROJECT_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e053-20261119/source-$mode-$commit
TARSHA=$(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | sha256sum | cut -d' ' -f1)
SOURCE_COMMIT=$(git -C "$root" rev-parse HEAD)
HDR
cat <<'BODY'
set -euo pipefail
cd /home/hpuvvad
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e053-20261119
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?; echo "$rc" > "$S/driver.exit"; touch "$S/FINISHED"; for ((i=0;i<90;i++)); do [[ -e "$S/RETRIEVED" ]] && break; sleep 2; done; exit "$rc"' EXIT
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -cx "$SLURM_JOB_NODELIST" "$S/launcher.log") == 1 ]]
mkdir "$PROJECT_ROOT"
base64 -d > "$S/source.tar.gz" <<'B64'
BODY
(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n | base64)
cat <<'BODY'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/CMakeLists.txt" && -f "$PROJECT_ROOT/scripts/e053/inner.sh" ]]
printf '%s\n' "$SOURCE_COMMIT" > "$S/source-commit.txt"
export OMPI_MCA_hwloc_base_binding_policy=none
PRE=; [[ -z "$PRE_NAME" ]] || PRE="$OUT/$PRE_NAME"
prun bash "$PROJECT_ROOT/scripts/e053/inner.sh" "$MODE" "$PROJECT_ROOT" "$S" "$PRE" > "$S/inner.log" 2>&1
(cd "$S" && find . -type f ! -name FINISHED ! -name RETRIEVED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
BODY
