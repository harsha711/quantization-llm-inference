#!/usr/bin/env bash
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e010_c4_20261107"
cycle_base() {
 bench_base
 base=("$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned"
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE")
}
