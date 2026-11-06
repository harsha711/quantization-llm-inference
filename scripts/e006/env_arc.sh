#!/usr/bin/env bash
# Source only after verifying allocation ownership, node and remaining time.
: "${PROJECT_ROOT:?Absolute E006 root}" "${SLURM_JOB_ID:?Verified allocation}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export E005_PYTHON=/tmp/e004-hpuvvad-20261106/bin/python
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export E005_CONTROL=/home/hpuvvad/csc548-w4a16-e005-20261106
export E004_CONTROL=/home/hpuvvad/csc548-w4a16-e004-20261106
export E006_RESULTS="$PROJECT_ROOT/results/e006_20261106"
export E006_WHEEL="$($E005_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')"
