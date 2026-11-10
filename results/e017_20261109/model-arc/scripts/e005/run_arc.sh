#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}" "${E005_PYTHON:?Isolated Python}" "${SLURM_JOB_ID:?Verified allocation}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
LIB=$($E005_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
exec prun timeout 900 "$E005_PYTHON" scripts/e005/bench.py \
 --source "$PROJECT_ROOT/external/pinned" --local results/e005_20261106/libe005.so \
 --wheel "$LIB" --bridge /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so "$@"
