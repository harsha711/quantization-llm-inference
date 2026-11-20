#!/usr/bin/env bash
# Local helper: emits the E061 batch script with the committed source tree embedded (as E058/E060).
#   bash scripts/e061/submit.sh | ssh arc-off-network sbatch --parsable --nodelist=c8
# One job, no prerequisite (predeclared): unit tests + mutation check, build/GPU checks and the all-row FP64
# validation run first and fail closed before any timing.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
paths=(CMakeLists.txt include src tests tools scripts/build.sh scripts/check_inblock.sh scripts/e004 scripts/e005 scripts/e008
       scripts/e010 scripts/e061 scripts/checks/floorcheck.py results/e010_c4_20261107/verified-inputs.sha256
       results/e050_20261118/full-c8-343308/series1.jsonl results/e050_20261118/full-c8-343308/series2.jsonl)
# Fail before printing anything unless every path is committed and clean (the job embeds git HEAD, not the
# working tree), and build the archive into a temp file first so a failure can never emit a truncated script.
git -C "$root" ls-files --error-unmatch "${paths[@]}" > /dev/null
git -C "$root" diff --quiet HEAD -- "${paths[@]}"
[[ -z $(git -C "$root" ls-files --others --exclude-standard -- "${paths[@]}") ]]
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
(cd "$root" && git archive HEAD "${paths[@]}" | gzip -n > "$tmp")
tarsha=$(sha256sum "$tmp" | cut -d' ' -f1)
b64=$(base64 < "$tmp")
[[ -n $b64 ]]
cat <<HDR
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e061
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/e061-%j.log
COMMIT=$(cd "$root" && git rev-parse HEAD)
TARSHA=$tarsha
HDR
cat <<'EOF2'
set -euo pipefail
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e061-20261120
mkdir -p "$OUT"; S="$OUT/full-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";(cd "$S" && find . -type f ! -name FINISHED ! -name "driver.*" ! -name all.sha256 | sort | xargs sha256sum > all.sha256);touch "$S/FINISHED";tar -C "$OUT" -czf "$HOME/e061-$(basename "$S").tgz" "$(basename "$S")";sha256sum "$HOME/e061-$(basename "$S").tgz" > "$HOME/e061-$(basename "$S").tgz.sha256";exit "$rc"' EXIT
echo "commit $COMMIT"
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
SRC=/tmp/e061-src-$SLURM_JOB_ID; rm -rf "$SRC"; mkdir "$SRC"
base64 -d > "$S/source.tar.gz" <<'B64'
EOF2
printf '%s\n' "$b64"
cat <<'EOF2'
B64
echo "$TARSHA  $S/source.tar.gz" | sha256sum -c
tar xzf "$S/source.tar.gz" -C "$SRC"
export OMPI_MCA_hwloc_base_binding_policy=none  # E041 attempt 1: mpirun bound the one rank to core 0
prun bash "$SRC/scripts/e061/inner.sh" "$SRC" "$S" > "$S/inner.log" 2>&1
echo STAGE-OK
EOF2
