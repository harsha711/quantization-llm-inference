#!/usr/bin/env bash
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e010_c4_20261107"
cycle_base() {
 bench_base
 base=("$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned"
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE")
}
representative_racecheck() {
 run racecheck compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=e005_marlin \
  "$E005_PYTHON" scripts/e005/bench.py --source "$E005_CONTROL/external/pinned" \
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
  --configs A --shapes 4096,11008 11008,4096 --ms 4 16 64 128 --seed 3 \
  --patterns mixed --policies no_flush --warmup 0 --iters 0 --repeats 3 --graph-check \
  --out "$R/racecheck.jsonl"
}
