#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
R=$E006_RESULTS
[[ ! -e "$R/original_square.log" ]]
base=("$E005_PYTHON" scripts/e006/repro.py --source "$E005_CONTROL/external/pinned"
 --local "$E005_CONTROL/results/e005_20261106/libe005.so" --wheel "$E006_WHEEL"
 --bridge "$E004_CONTROL/results/e004_20261106/libe004.so")
# Cheap actual-driver correctness prerequisite before instrumentation.
for size in square small; do
 dims=(--k 4096 --n 4096)
 [[ $size == square ]] || dims=(--k 256 --n 2048)
 set +e
 prun timeout 300 compute-sanitizer --tool racecheck --racecheck-report all --error-exitcode 99 \
  --kernel-name kns=e005_marlin \
  "${base[@]}" "${dims[@]}" --out "$R/original_${size}.json" > "$R/original_${size}.log" 2>&1
 rc=$?
 set -e
 echo "$rc" > "$R/original_${size}.exit"
 [[ $rc == 99 ]] || exit 1
done
