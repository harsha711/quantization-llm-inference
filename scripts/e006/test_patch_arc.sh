#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
R=$E006_RESULTS
for variant in control patched; do
 [[ ! -e "$R/${variant}_square.log" ]]
 set +e
 prun timeout 300 compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=e005_marlin \
  "$E005_PYTHON" scripts/e006/repro.py --source "$E005_CONTROL/external/pinned" \
  --local "$R/$variant/libe005.so" --wheel "$E006_WHEEL" --bridge "$E004_CONTROL/results/e004_20261106/libe004.so" \
  --out "$R/${variant}_square.json" > "$R/${variant}_square.log" 2>&1
 rc=$?
 set -e
 echo "$rc" > "$R/${variant}_square.exit"
 expected=99
 [[ $variant == control ]] || expected=0
 [[ $rc == "$expected" ]] || exit 1
done
