#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Absolute project}"
: "${E004_PYTHON:?Isolated Python}"
: "${SLURM_JOB_ID:?Verified allocation}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R=results/e004_20261106
mkdir "$R/ncu"
LIB="$("$E004_PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')"
metrics=(
 gpu__time_duration.sum launch__grid_size launch__block_size launch__registers_per_thread
 launch__shared_mem_per_block_static launch__shared_mem_per_block_dynamic
 launch__occupancy_limit_blocks launch__occupancy_limit_warps launch__waves_per_multiprocessor
 sm__warps_active.avg.pct_of_peak_sustained_active smsp__warps_eligible.avg.per_cycle_active
 smsp__issue_active.avg.pct_of_peak_sustained_active smsp__inst_executed.sum
 sm__inst_executed_pipe_tensor.sum sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
 smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
 smsp__sass_thread_inst_executed_op_integer_pred_on.sum
 smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
 sm__throughput.avg.pct_of_peak_sustained_elapsed dram__throughput.avg.pct_of_peak_sustained_elapsed
 dram__bytes.sum dram__bytes_op_read.sum dram__bytes_op_write.sum lts__t_bytes.sum
)
for metric in "${metrics[@]}"; do
 if [[ "$metric" != launch__* ]]; then
  zgrep -Fq "$metric " "$R/ncu-metrics.txt.gz" || { echo "Missing metric $metric" >&2; exit 1; }
 fi
done
list=$(IFS=,; echo "${metrics[*]}")
for M in 8 16 64; do
 tag="marlin_M${M}_reused_kernel_replay"
 prun ncu --clock-control none --cache-control none --profile-from-start off \
  --kernel-name-base demangled --kernel-name 'regex:marlin::Marlin<' --launch-count 1 \
  --metrics "$list" --csv -o "$R/ncu/$tag" \
  "$E004_PYTHON" scripts/e004/profile.py --source "$PROJECT_ROOT/external/vllm" \
  --library "$LIB" --bridge "$R/libe004.so" --M "$M" --warmup 10 \
  > "$R/ncu/$tag.log" 2> "$R/ncu/$tag.stderr"
 ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv > "$R/ncu/${tag}_metrics.csv"
done
