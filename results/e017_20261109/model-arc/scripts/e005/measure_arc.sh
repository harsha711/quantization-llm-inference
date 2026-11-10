#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?}" "${E005_PYTHON:?}" "${SLURM_JOB_ID:?}"
cd "$PROJECT_ROOT"
R=results/e005_20261106
[[ -e "$R/VALIDATED" ]]
sha256sum -c "$R/validated-library.sha256"
sha256sum scripts/e005/*.* scripts/e004/bench.py "$R/libe005.so" > "$R/measurement-source.sha256"
clock() { nvidia-smi --query-gpu=timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
for series in 1 2; do
 clock > "$R/series${series}-clocks-before.csv"
 bash scripts/e005/run_arc.sh --out "$R/series${series}.jsonl" --ms 1 8 --shapes 4096,4096 \
  --configs A B C --warmup 10 --iters 100 --balance "$series" > "$R/series${series}.log" 2>&1
 clock > "$R/series${series}-clocks-after.csv"
 bash scripts/e005/run_arc.sh --out "$R/fallback${series}.jsonl" --ms 16 64 --shapes 4096,4096 \
  --configs A B C --warmup 10 --iters 100 --balance "$series" > "$R/fallback${series}.log" 2>&1
 clock > "$R/fallback${series}-clocks-after.csv"
done
# Candidate code/dispatch is fixed before inspecting held-out seed-2 points.
bash scripts/e005/run_arc.sh --out "$R/heldout.jsonl" --ms 2 4 --seed 2 \
 --shapes 4096,4096 4096,11008 11008,4096 --configs A B C \
 --warmup 10 --iters 100 --balance 1 > "$R/heldout.log" 2>&1
clock > "$R/heldout-clocks-after.csv"
# Wheel is diagnostic only, in a process which never dlopens the local library.
bash scripts/e005/run_arc.sh --out "$R/wheel_diagnostic.jsonl" --ms 1 8 --configs wheel \
 --shapes 4096,4096 --warmup 10 --iters 100 --trace > "$R/wheel_diagnostic.log" 2>&1
clock > "$R/wheel-clocks-after.csv"
