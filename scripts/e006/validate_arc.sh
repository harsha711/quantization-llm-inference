#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e006/env_arc.sh"
R=$E006_RESULTS
[[ $(cat "$R/patched_square.exit") == 0 ]]
grep -q 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' "$R/patched_square.log"
base=("$E005_PYTHON" scripts/e005/bench.py --source "$E005_CONTROL/external/pinned"
 --local "$R/patched/libe005.so" --wheel "$E006_WHEEL" --bridge "$E004_CONTROL/results/e004_20261106/libe004.so"
 --configs A B C --warmup 0 --iters 0 --policies no_flush --graph-check)
run_check() {
 local label=$1
 shift
 [[ ! -e "$R/$label.log" ]]
 set +e
 (
  set -x
  prun timeout 900 "$@"
 ) > "$R/$label.log" 2>&1
 local rc=$?
 set -e
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
run_check validation_small "${base[@]}" --out "$R/validation_small.jsonl" --ms 1 5 8 16 64 \
 --shapes 256,128 256,2048 --patterns mixed zero_a zero_w --repeats 5
run_check validation_main "${base[@]}" --out "$R/validation_main.jsonl" --ms 1 5 8 16 64 \
 --shapes 4096,4096 4096,11008 11008,4096 --repeats 5
for tool in memcheck racecheck synccheck; do
 run_check "${tool}_small" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "${base[@]}" --out "$R/${tool}_small.jsonl" --ms 1 5 8 --shapes 256,2048 \
  --patterns mixed zero_a zero_w --repeats 2
 run_check "${tool}_main" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "${base[@]}" --out "$R/${tool}_main.jsonl" --ms 8 16 64 \
  --shapes 4096,4096 4096,11008 11008,4096 --repeats 1
done
# This task never creates E005's VALIDATED or releases performance work.
