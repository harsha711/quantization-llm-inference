#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?Absolute project path}"
cd "$PROJECT_ROOT"
R=results/e004_20261106
# Boundary M8->M16 and near cuBLAS crossover M64, square weights.
for repeat in 1 2; do
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate --format=csv > "$R/repeat${repeat}_gpu_before.csv"
  bash scripts/e004/run_arc.sh --out "$R/repeat${repeat}.jsonl" --shapes 4096,4096 --ms 1 8 16 64 > "$R/repeat${repeat}.log" 2>&1
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate --format=csv > "$R/repeat${repeat}_gpu_after.csv"
done
# Same points/selectors/sample counts in both launch-overhead diagnostics.
for method in graph eager; do
  args=()
  if [[ "$method" == eager ]]; then args+=(--eager); fi
  bash scripts/e004/run_arc.sh --out "$R/${method}_diagnostic.jsonl" --shapes 4096,4096 --ms 1 32 \
    --kernels marlin inblock_specialized cublas_dq "${args[@]}" > "$R/${method}_diagnostic.log" 2>&1
done
