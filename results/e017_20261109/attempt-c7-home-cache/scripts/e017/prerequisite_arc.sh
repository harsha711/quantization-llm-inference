#!/usr/bin/env bash
# Executed inside the cheap bootstrap allocation after isolated staging/runtime.
set -euo pipefail
source "$PROJECT_ROOT/scripts/e017/env_arc.sh"
allocation prerequisite
run quantizer-check "$E017_PYTHON" scripts/e017/check_quantize.py
run inference-local-check "$E017_PYTHON" scripts/e017/check_infer.py
run capture "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode capture
for m in 1 128 2048; do
 run "one-layer-M$m" "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode check --m "$m"
done
for tool in memcheck synccheck racecheck; do
 ms=(1 128 2048);if [[ "$tool" == racecheck ]];then ms=(1 128);fi
 for m in "${ms[@]}";do
  run "one-layer-$tool-M$m" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode check --m "$m" --suffix "-$tool" --no-trace
 done
done
clock > "$R/prerequisite-clocks.csv"
touch "$R/ONE_LAYER_CAPTURED"
