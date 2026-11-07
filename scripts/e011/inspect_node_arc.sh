#!/usr/bin/env bash
# Metadata-only prerequisite for an alternative node, no benchmark execution.
set -euo pipefail
: "${PROJECT_ROOT:?Export an absolute project path}"
[[ "$PROJECT_ROOT" == /* && -d "$PROJECT_ROOT/scripts/e011" ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
scontrol show job "$SLURM_JOB_ID"
prun /bin/hostname
prun nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv
ncu --version
prun ncu --query-metrics --metrics gpu__time_duration --query-metrics-mode all
