#!/usr/bin/env bash
# Correctness tests + the M in {1,8,32,128}, K=N=4096 baseline sweep.
# Run on a GPU node after scripts/build.sh. Writes results/baseline_*.csv and logs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/ncu
export PATH="/usr/local/cuda/bin:${PATH}"

META=results/baseline_meta.txt
{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
  echo "--- cudaDeviceProp L2 is printed by w4a16_bench ---"
  nvidia-smi -q | grep -i -E "Product Name|Driver Version|CUDA Version|L2 Cache" || true
  nvcc --version
  ncu --version | tail -1
  cmake --version | head -1
  echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
  grep RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null || true
} | tee "$META"

./build/test_cpu | tee results/test_cpu.log

B=./build/w4a16_bench
CSV=results/baseline_c7.csv
LOG=results/baseline_c7.log
rm -f "$CSV"
: > "$LOG"

run() { echo "+ $*" | tee -a "$LOG"; "$@" | tee -a "$LOG"; }

run "$B" --M 3 --N 1000 --K 384 --group 128 --seed 1 --warmup 2 --iters 5 --csv "$CSV"
run "$B" --M 8 --N 4096 --K 4096 --group 4096 --seed 1 --warmup 2 --iters 5 --csv "$CSV"

for M in 1 8 32 128; do
  run "$B" --M "$M" --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters 100 --csv "$CSV"
done

# Same M=1 naive launch with L2 left warm, to test whether the flush changes latency.
run "$B" --M 1 --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters 100 --kernels naive --no-flush-l2 --csv "$CSV"

echo "baseline csv: $CSV" | tee -a "$LOG"
