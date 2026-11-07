#!/usr/bin/env bash
# Reuse the verified launcher/runtime policy, with a separate output directory.
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
R="$PROJECT_ROOT/results/e008_c10_20261106"
cycle_base() {
 bench_base
 base=("$E005_PYTHON" scripts/e008/bench.py --source "$E005_CONTROL/external/pinned"
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE")
}
