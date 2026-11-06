#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}" "${SLURM_JOB_ID:?Verified allocation}" "${E005_PYTHON:?Isolated Python}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR="$PROJECT_ROOT/results/e005_20261106"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/allocation.txt"
prun /bin/hostname > "$RUN_DIR/launcher.log"
[[ $(grep -xc "$(hostname)" "$RUN_DIR/launcher.log") == 1 ]]
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
 ncu --version
 nvidia-smi -q
 env | sort | grep -E '^(SLURM|OMP|OMPI|PROJECT_ROOT|E005_PYTHON)'
} > "$RUN_DIR/metadata.txt" 2>&1
[[ ! -e external/pinned ]]
tar -xzf "$RUN_DIR/pinned-source.tar.gz" -C external
"$E005_PYTHON" scripts/e005/prepare_source.py external/pinned external/local
cmp external/local/e005-host.patch "$RUN_DIR/e005-host.patch"
prun "$E005_PYTHON" scripts/e004/check_runtime.py > "$RUN_DIR/runtime-prun.log" 2>&1
prun /bin/bash scripts/e005/build.sh > "$RUN_DIR/build.log" 2>&1
