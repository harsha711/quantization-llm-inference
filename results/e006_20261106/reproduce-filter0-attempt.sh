#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
R=$E006_RESULTS
[[ ! -e "$R/original_square.log" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
prun /bin/hostname > "$R/launcher.log"
[[ $(grep -xc "$(hostname)" "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 hostname
 lscpu
 numactl --hardware
 cat /proc/self/status
 nvcc --version
 gcc --version
 mpirun --version
 cmake --version
 compute-sanitizer --version
 nvidia-smi -q
 env | sort | grep -E '^(SLURM|OMP|OMPI|PROJECT_ROOT|E005_PYTHON)'
 sha256sum "$E005_CONTROL/results/e005_20261106/libe005.so" "$E006_WHEEL" "$E004_CONTROL/results/e004_20261106/libe004.so"
} > "$R/metadata.txt" 2>&1
prun "$E005_PYTHON" scripts/e004/check_runtime.py > "$R/runtime-prun.log" 2>&1
base=("$E005_PYTHON" scripts/e006/repro.py --source "$E005_CONTROL/external/pinned"
 --local "$E005_CONTROL/results/e005_20261106/libe005.so" --wheel "$E006_WHEEL"
 --bridge "$E004_CONTROL/results/e004_20261106/libe004.so")
# Cheap actual-driver correctness prerequisite before instrumentation.
prun timeout 300 "${base[@]}" --k 256 --n 2048 --out "$R/prerequisite.json" > "$R/prerequisite.log" 2>&1
for size in square small; do
 dims=(--k 4096 --n 4096)
 [[ $size == square ]] || dims=(--k 256 --n 2048)
 set +e
 prun timeout 300 compute-sanitizer --tool racecheck --racecheck-report all --error-exitcode 99 \
  --kernel-name kns=e005_marlin --racecheck-filter-cluster-id 0 \
  "${base[@]}" "${dims[@]}" --out "$R/original_${size}.json" > "$R/original_${size}.log" 2>&1
 rc=$?
 set -e
 echo "$rc" > "$R/original_${size}.exit"
 [[ $rc == 99 ]] || exit 1
done
