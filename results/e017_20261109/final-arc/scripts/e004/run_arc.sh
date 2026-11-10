#!/usr/bin/env bash
# Helper for the real shared path. Reverify ownership/time before invoking.
set -euo pipefail
: "${PROJECT_ROOT:?Absolute isolated E004 project}"
: "${SLURM_JOB_ID:?Verified running user allocation}"
: "${E004_PYTHON:?Isolated Python}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
RUN_DIR="$PROJECT_ROOT/results/e004_20261106"
LIB="$("$E004_PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')"
sha256sum -c "$RUN_DIR/check.sha256"
exec prun "$E004_PYTHON" scripts/e004/bench.py --source "$PROJECT_ROOT/external/vllm" \
  --library "$LIB" --bridge "$RUN_DIR/libe004.so" "$@"
