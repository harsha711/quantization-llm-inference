#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute staged project path}" "${SLURM_JOB_ID:?Live allocation}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e013/bench.py" ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export E013_PYTHON=/tmp/e005-r2-hpuvvad-20261106/bin/python
export E013_SOURCE="$PROJECT_ROOT/external/e013"
export E013_RESULTS="$PROJECT_ROOT/results/e013_20261108"
export E004_CONTROL=/home/hpuvvad/csc548-w4a16-e004-20261106
export E006_CONTROL=/home/hpuvvad/csc548-w4a16-e006-20261106
export E013_BRIDGE="$E004_CONTROL/results/e004_20261106/libe004.so"
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$E013_RESULTS"
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
bench_base() {
 E013_WHEEL=$("$E013_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
 export E013_WHEEL
 base=("$E013_PYTHON" scripts/e013/bench.py --source "$E013_SOURCE"
  --local "$R/libe005.so" --wheel "$E013_WHEEL" --bridge "$E013_BRIDGE")
}
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
