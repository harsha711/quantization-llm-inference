#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${E005_PYTHON:?}" "${SLURM_JOB_ID:?}"
cd "$PROJECT_ROOT"
R=results/e005_20261106
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
prun timeout 900 bash /home/hpuvvad/csc548-w4a16-e004-20261106/scripts/check_inblock.sh > "$R/project-check.log" 2>&1
bash scripts/e005/run_arc.sh --out "$R/validation_small.jsonl" --ms 1 2 4 5 8 --shapes 256,128 256,2048 \
 --configs A B C --patterns mixed zero_a zero_w --warmup 0 --iters 0 --repeats 5 --graph-check > "$R/validation_small.log" 2>&1
bash scripts/e005/run_arc.sh --out "$R/validation_main.jsonl" --ms 1 2 4 5 8 16 64 \
 --shapes 4096,4096 4096,11008 11008,4096 --configs A B C --warmup 0 --iters 0 --repeats 5 --graph-check > "$R/validation_main.log" 2>&1
LIB=$($E005_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
for tool in memcheck racecheck synccheck; do
 # Filter only the external local Marlin body. The repacking/fill kernels and
 # Torch runtime are outside this sanitizer coverage.
 prun timeout 600 compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "$E005_PYTHON" scripts/e005/bench.py --source external/pinned --local "$R/libe005.so" --wheel "$LIB" \
  --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so \
  --out "$R/sanitizer_${tool}.jsonl" --ms 1 5 8 --shapes 256,2048 --configs A B C \
  --warmup 0 --iters 0 --repeats 2 --graph-check > "$R/sanitizer_${tool}.log" 2>&1
done
# The rectangular 72-CTA case reaches lock 70 and the largest scratch offset.
prun timeout 600 compute-sanitizer --tool memcheck --error-exitcode 99 --kernel-name kns=e005_marlin \
 "$E005_PYTHON" scripts/e005/bench.py --source external/pinned --local "$R/libe005.so" --wheel "$LIB" \
 --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so \
 --out "$R/sanitizer_memcheck_rect.jsonl" --ms 5 --shapes 4096,11008 --configs A B C \
 --warmup 0 --iters 0 --repeats 2 --graph-check > "$R/sanitizer_memcheck_rect.log" 2>&1
sha256sum "$R/libe005.so" > "$R/validated-library.sha256"
touch "$R/VALIDATED"
