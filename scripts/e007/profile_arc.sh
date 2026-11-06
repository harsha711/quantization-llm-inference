#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e007/env_arc.sh"
[[ $(cat "$R/candidate_heldout.exit") == 0 && $(cat "$R/control_heldout.exit") == 0 ]]
sha256sum -c "$R/validated-inputs.sha256" > "$R/profile-input-check.log"
mkdir "$R/ncu"
run metric-query ncu --query-metrics --query-metrics-mode all
metrics=(
 gpu__time_duration.sum launch__grid_size launch__block_size launch__registers_per_thread
 launch__shared_mem_per_block_static launch__shared_mem_per_block_dynamic
 launch__occupancy_limit_blocks launch__occupancy_limit_warps
 sm__ctas_active.avg.per_cycle_active sm__warps_active.avg.pct_of_peak_sustained_active
 smsp__warps_eligible.avg.per_cycle_active smsp__issue_active.avg.pct_of_peak_sustained_active
 smsp__inst_executed.sum smsp__inst_executed_op_ldsm.sum
 smsp__inst_executed_op_shared_ld.sum smsp__inst_executed_op_shared_ld_pred_on_any.sum
 sm__inst_executed_pipe_tensor.sum sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
 smsp__sass_thread_inst_executed_op_integer_pred_on.sum
 smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed
 smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
 dram__bytes.sum dram__bytes_op_read.sum dram__bytes_op_write.sum lts__t_bytes.sum
)
for metric in "${metrics[@]}"; do
 if [[ "$metric" != launch__* ]]; then
  grep -Fq "$metric " "$R/metric-query.log" || { echo "Missing metric $metric" >&2; exit 1; }
 fi
done
list=$(IFS=,; echo "${metrics[*]}")
printf '%s\n' "${metrics[@]}" > "$R/profile-metrics.txt"
for variant in control candidate; do
 tag="${variant}_M8_reused_kernel_replay"
 clock > "$R/ncu/$tag-clocks-before.csv"
 run "${variant}_profile" ncu --clock-control none --cache-control none --profile-from-start off \
  --kernel-name-base demangled --kernel-name 'regex:e005_marlin::Marlin<' --launch-count 1 \
  --metrics "$list" --csv -o "$R/ncu/$tag" \
  "${base[@]}" --variant "$variant" --out "$R/${variant}_profile.jsonl" --ms 8 --shapes 4096,4096 \
  --configs A --warmup 10 --iters 0 --policies no_flush --profile
 ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv --print-kernel-base mangled > "$R/ncu/${tag}_metrics.csv"
 clock > "$R/ncu/$tag-clocks-after.csv"
done
sha256sum scripts/e007/* scripts/e005/bench.py scripts/e004/bench.py > "$R/profile-tooling.sha256"
