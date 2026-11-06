#!/usr/bin/env bash
# Same reduced shape after hazard+analysis reporting exceeded its timeout.
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
R=$E006_RESULTS
[[ $(cat "$R/original_small.exit") == 124 ]]
[[ ! -e "$R/original_small_analysis.log" ]]
set +e
prun timeout 300 compute-sanitizer --tool racecheck --racecheck-report analysis --error-exitcode 99 --kernel-name kns=e005_marlin \
 "$E005_PYTHON" scripts/e006/repro.py --source "$E005_CONTROL/external/pinned" \
 --local "$E005_CONTROL/results/e005_20261106/libe005.so" --wheel "$E006_WHEEL" \
 --bridge "$E004_CONTROL/results/e004_20261106/libe004.so" --k 256 --n 2048 \
 --out "$R/original_small_analysis.json" > "$R/original_small_analysis.log" 2>&1
rc=$?
set -e
echo "$rc" > "$R/original_small_analysis.exit"
[[ $rc == 99 ]]
