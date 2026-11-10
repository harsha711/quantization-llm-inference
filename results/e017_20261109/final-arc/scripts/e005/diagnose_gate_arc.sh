#!/usr/bin/env bash
# Bounded diagnostics after racecheck stopped validation. No performance series.
set -euo pipefail
: "${PROJECT_ROOT:?}" "${E005_PYTHON:?}" "${SLURM_JOB_ID:?}"
cd "$PROJECT_ROOT"
R=results/e005_20261106
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
LIB=$($E005_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
base=(--source external/pinned --local "$R/libe005.so" --wheel "$LIB"
 --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so
 --ms 8 --shapes 4096,4096 --warmup 0 --iters 0 --repeats 1 --policies no_flush)
# Expected failures are captured explicitly for comparison. They cannot create
# the VALIDATED sentinel and never release measurement work.
for name in A wheel; do
 set +e
 prun timeout 300 compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=marlin \
  "$E005_PYTHON" scripts/e005/bench.py "${base[@]}" --configs "$name" --out "$R/race_diagnostic_${name}.jsonl" \
  > "$R/race_diagnostic_${name}.log" 2>&1
 status=$?
 set -e
 echo "$status" > "$R/race_diagnostic_${name}.exit"
 [[ "$status" == 0 || "$status" == 99 ]] || exit "$status"
done
prun timeout 600 compute-sanitizer --tool synccheck --error-exitcode 99 --kernel-name kns=e005_marlin \
 "$E005_PYTHON" scripts/e005/bench.py --source external/pinned --local "$R/libe005.so" --wheel "$LIB" \
 --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so \
 --out "$R/sanitizer_synccheck.jsonl" --ms 1 5 8 --shapes 256,2048 --configs A B C \
 --warmup 0 --iters 0 --repeats 2 --graph-check > "$R/sanitizer_synccheck.log" 2>&1
prun timeout 600 compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=e005_marlin \
 "$E005_PYTHON" scripts/e005/bench.py --source external/pinned --local "$R/libe005.so" --wheel "$LIB" \
 --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so \
 --out "$R/sanitizer_memcheck_rect.jsonl" --ms 5 --shapes 4096,11008 --configs A B C \
 --warmup 0 --iters 0 --repeats 2 --graph-check > "$R/sanitizer_memcheck_rect.log" 2>&1
# Resolve initial mixed-library profiler naming without measuring latency.
bash scripts/e005/run_arc.sh --out "$R/wheel_selection.jsonl" --ms 8 --configs wheel \
 --shapes 4096,4096 --warmup 0 --iters 0 --repeats 2 --trace > "$R/wheel_selection.log" 2>&1
