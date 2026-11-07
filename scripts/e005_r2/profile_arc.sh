#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
[[ -e "$R/VALIDATION_PASSED" && $(cat "$R/metric-query.exit") == 0 ]]
bench_base
mapfile -t shape < <("$E005_PYTHON" scripts/e005_r2/profile_shape.py "$R/analysis.json")
[[ ${#shape[@]} == 2 ]]
sha256sum -c "$R/verified-inputs.sha256" > "$R/profile-input-check.log"
mkdir "$R/ncu"
metrics=(
 gpu__time_duration.sum launch__grid_size launch__block_size launch__registers_per_thread
 launch__shared_mem_per_block_static launch__shared_mem_per_block_dynamic
 launch__occupancy_limit_blocks launch__occupancy_limit_warps launch__waves_per_multiprocessor
 sm__ctas_active.avg.per_cycle_active sm__ctas_active.avg.per_cycle_elapsed
 sm__warps_active.avg.pct_of_peak_sustained_active smsp__warps_eligible.avg.per_cycle_active
 smsp__issue_active.avg.pct_of_peak_sustained_active smsp__inst_executed.sum
 sm__inst_executed_pipe_tensor.sum sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
 smsp__sass_thread_inst_executed_op_integer_pred_on.sum
 smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_no_instruction.avg.pct_of_peak_sustained_elapsed
 sm__throughput.avg.pct_of_peak_sustained_elapsed dram__throughput.avg.pct_of_peak_sustained_elapsed
 dram__bytes.sum dram__bytes_op_read.sum dram__bytes_op_write.sum lts__t_bytes.sum
 lts__throughput.avg.pct_of_peak_sustained_elapsed
)
for metric in "${metrics[@]}"; do
 if [[ "$metric" != launch__* ]]; then
  grep -Fq "$metric " "$R/metric-query.log" || { echo "Missing metric $metric" >&2; exit 1; }
 fi
done
list=$(IFS=,; echo "${metrics[*]}")
printf '%s\n' "${metrics[@]}" > "$R/profile-metrics.txt"
sha256sum scripts/e005_r2/* scripts/e005/bench.py scripts/e004/bench.py > "$R/profile-tooling.sha256"
for config in A C; do
 tag="${config}_M4_K${shape[0]}_N${shape[1]}_reused_kernel_replay"
 clock > "$R/ncu/$tag-clocks-before.csv"
 run "profile-$config" ncu --clock-control none --cache-control none --profile-from-start off \
  --kernel-name-base demangled --kernel-name 'regex:e005_marlin::Marlin<' --launch-count 1 \
  --metrics "$list" --csv -o "$R/ncu/$tag" \
  "${base[@]}" --configs "$config" --out "$R/profile-$config.jsonl" --shapes "${shape[0]},${shape[1]}" \
  --warmup 10 --iters 0 --policies no_flush --profile
 ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv > "$R/ncu/${tag}_metrics.csv"
 ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv --print-kernel-base mangled > "$R/ncu/${tag}_mangled.csv"
 clock > "$R/ncu/$tag-clocks-after.csv"
done
