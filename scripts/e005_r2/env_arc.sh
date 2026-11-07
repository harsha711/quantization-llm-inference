#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute E005-R2 project path}" "${SLURM_JOB_ID:?Verified allocation}"
[[ "$PROJECT_ROOT" == /* ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export E005_PYTHON=/tmp/e005-r2-hpuvvad-20261106/bin/python
export E005_CONTROL=/home/hpuvvad/csc548-w4a16-e005-20261106
export E006_CONTROL=/home/hpuvvad/csc548-w4a16-e006-20261106
export E004_CONTROL=/home/hpuvvad/csc548-w4a16-e004-20261106
export E005_LOCAL="$E006_CONTROL/results/e006_20261106/patched/libe005.so"
export E005_BRIDGE="$E004_CONTROL/results/e004_20261106/libe004.so"
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$PROJECT_ROOT/results/e005_r2_c10_20261106"
run() {
 local label=$1
 shift
 [[ ! -e "$R/$label.log" && ! -e "$R/$label.exit" ]]
 local rc=0
 (set -x; prun timeout 1200 "$@") > "$R/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
clock() { nvidia-smi --query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
bench_base() {
 export E005_WHEEL="$($E005_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')"
 base=("$E005_PYTHON" scripts/e005/bench.py --source "$E005_CONTROL/external/pinned"
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" --configs A C
  --ms 4 --seed 2 --graph-check --repeats 3)
}
