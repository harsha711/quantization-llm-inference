#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute E017 project root}" "${SLURM_JOB_ID:?Live allocation}"
[[ "$PROJECT_ROOT" == /mnt/beegfs/hpuvvad/csc548-w4a16-e017-r1-20261109 ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export E017_PYTHON=/tmp/e017-hpuvvad-20261109/bin/python
export E017_LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
export E017_SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
export E017_BRIDGE=/home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so
export E017_WHEEL="$PROJECT_ROOT/external/_C_stable_libtorch.abi3.so"
export TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/inductor"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg" TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/cache/torch-extensions"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export R="$PROJECT_ROOT/results/e017_20261109"
allocation() {
 scontrol show job "$SLURM_JOB_ID" > "$R/$1-allocation.txt"
 grep -q 'UserId=hpuvvad' "$R/$1-allocation.txt"
 grep -q 'JobState=RUNNING' "$R/$1-allocation.txt"
 grep -q "NodeList=$SLURM_JOB_NODELIST" "$R/$1-allocation.txt"
 [[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
 [[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
 [[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) == '12.0' ]]
}
run() {
 local label=$1 rc=0;shift
 [[ ! -e "$R/$label.log" && ! -e "$R/$label.exit" ]]
 (set -x;prun timeout 1800 "$@") > "$R/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
