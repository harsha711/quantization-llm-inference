#!/usr/bin/env bash
# Nsight Compute on naive (M=1 and M=128) and dequant (M=1).
# dram__bytes_read.sum is n/a on the RTX 5060 Ti; use dram__bytes_op_*.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/ncu
export PATH="/usr/local/cuda/bin:${PATH}"

B=./build/w4a16_bench
METRICS=gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread,launch__grid_size,launch__block_size,launch__occupancy_limit_blocks,launch__occupancy_limit_warps,launch__waves_per_multiprocessor,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,lts__t_sectors_aperture_device_lookup_hit.sum,lts__t_sectors_aperture_device_lookup_miss.sum,smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed

profile() {
  local tag=$1; shift
  echo "== ncu $tag: $*"
  ncu --cache-control all --kernel-name regex:"w4a16_naive_kernel|dequant_w4a16_kernel" \
      --launch-count 1 --csv --metrics "$METRICS" \
      -o "results/ncu/${tag}" \
      "$@" > "results/ncu/${tag}.log" 2> "results/ncu/${tag}.stderr"
  ncu --import "results/ncu/${tag}.ncu-repz" --page raw --csv > "results/ncu/${tag}_metrics.csv"
}

profile naive_M1_cold "$B" --M 1 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels naive --nocheck --no-flush-l2
profile naive_M128 "$B" --M 128 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels naive --nocheck --no-flush-l2
profile dequant_M1 "$B" --M 1 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels dequant --nocheck --no-flush-l2
echo "ncu reports under results/ncu/"
