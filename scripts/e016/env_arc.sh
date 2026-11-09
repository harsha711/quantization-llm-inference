#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute E016 root}" "${SLURM_JOB_ID:?Live allocation}"
[[ "$PROJECT_ROOT" == /mnt/beegfs/hpuvvad/csc548-w4a16-e016-c9-20261109 ]]
[[ -f "$PROJECT_ROOT/scripts/e016/run.py" ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export E016_PYTHON=/tmp/e015-hpuvvad-20261109/bin/python
export HF_HOME="$PROJECT_ROOT/cache/huggingface" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export R="$PROJECT_ROOT/results/e016_20261109"
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
 local label=$1 rc=0
 shift
 [[ ! -e "$R/$label.log" && ! -e "$R/$label.exit" ]]
 (set -x; prun timeout 1800 "$@") > "$R/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
