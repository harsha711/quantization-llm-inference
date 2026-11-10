#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute E014 project root}" "${SLURM_JOB_ID:?Live allocation}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e014/setup_arc.sh" ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export E014_PYTHON=/tmp/e014-hpuvvad-20261108/bin/python
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/inductor"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export R="$PROJECT_ROOT/results/e014_20261108"
allocation() {
 scontrol show job "$SLURM_JOB_ID" > "$R/$1-allocation.txt"
 grep -q 'UserId=hpuvvad' "$R/$1-allocation.txt"
 grep -q 'JobState=RUNNING' "$R/$1-allocation.txt"
 grep -q 'NodeList=c4' "$R/$1-allocation.txt"
 [[ $(hostname) == c4 ]]
 [[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
 [[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) == '12.0' ]]
}
run() {
 local label=$1
 shift
 [[ ! -e "$R/$label.log" && ! -e "$R/$label.exit" ]]
 local rc=0
 (set -x; prun timeout 1800 "$@") > "$R/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
