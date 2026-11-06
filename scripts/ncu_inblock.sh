#!/usr/bin/env bash
# Separate Nsight cache invalidation experiment, without the application's flush.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${RUN_DIR:?Set RUN_DIR to the completed experiment directory}"
sha256sum -c "$RUN_DIR/check.sha256"
mkdir "$RUN_DIR/ncu"
if [[ ! -f "$RUN_DIR/metric_names.txt" ]]; then
  ncu --query-metrics --query-metrics-mode all > "$RUN_DIR/metric_names.txt"
fi
gzip -c "$RUN_DIR/metric_names.txt" > "$RUN_DIR/ncu/metric_names.txt.gz"
metrics=(
  gpu__time_duration.sum
  launch__grid_size launch__block_size launch__registers_per_thread
  launch__shared_mem_per_block_static launch__shared_mem_per_block_dynamic
  launch__occupancy_limit_blocks launch__occupancy_limit_warps
  launch__waves_per_multiprocessor
  sm__warps_active.avg.pct_of_peak_sustained_active
  smsp__warps_eligible.avg.per_cycle_active
  smsp__issue_active.avg.pct_of_peak_sustained_active
  smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed
  smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
  smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed
  smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_elapsed
  smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes.sum dram__bytes_op_read.sum dram__bytes_op_write.sum
  lts__t_bytes.sum
)
for metric in "${metrics[@]}"; do
  # Launch attributes are collected by Nsight and are absent from its HW metric query.
  if [[ "$metric" != launch__* ]]; then
    grep -Fq "$metric " "$RUN_DIR/metric_names.txt" || {
      echo "Unavailable metric: $metric" >&2; exit 1;
    }
  fi
done
metric_list=$(IFS=,; echo "${metrics[*]}")
set -x
for M in 1 128; do
  for kernel in naive inblock; do
    tag="${kernel}_M${M}_ncu_invalidated"
    ncu --cache-control all --kernel-name "regex:w4a16_${kernel}_kernel" \
      --launch-count 1 --csv --metrics "$metric_list" -o "$RUN_DIR/ncu/$tag" \
      ./build/w4a16_bench --M "$M" --N 4096 --K 4096 --group 128 --seed 1 \
      --warmup 0 --iters 1 --kernels "$kernel" --nocheck --no-flush-l2 \
      > "$RUN_DIR/ncu/$tag.log" 2> "$RUN_DIR/ncu/$tag.stderr"
    ncu --import "$RUN_DIR/ncu/$tag.ncu-repz" --page raw --csv \
      > "$RUN_DIR/ncu/${tag}_metrics.csv"
  done
done
