#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e007/env_arc.sh"
if [[ ${1:-smoke} == smoke ]]; then
 run build-check "$E005_PYTHON" scripts/e007/check.py "$R" --stage build --e006 "$E006_CONTROL/results/e006_20261106"
 for variant in control candidate; do
  run "${variant}_smoke" "${base[@]}" --variant "$variant" --out "$R/${variant}_smoke.jsonl" --ms 1 5 8 \
   --shapes 256,2048 --configs A B C --patterns mixed zero_a zero_w --policies no_flush --repeats 3
 done
 run smoke-check "$E005_PYTHON" scripts/e007/check.py "$R" --stage smoke --e006 "$E006_CONTROL/results/e006_20261106"
 exit 0
fi
[[ $1 == full && $(cat "$R/smoke-check.exit") == 0 ]]
for variant in control candidate; do
 run "${variant}_small" "${base[@]}" --variant "$variant" --out "$R/${variant}_small.jsonl" --ms 1 5 8 16 64 \
  --shapes 256,128 256,2048 --configs A B C --patterns mixed zero_a zero_w --policies no_flush --repeats 5
 run "${variant}_main" "${base[@]}" --variant "$variant" --out "$R/${variant}_main.jsonl" --ms 1 5 8 16 64 \
  --shapes 4096,4096 4096,11008 11008,4096 --configs A B C --policies no_flush --repeats 5
done
for tool in memcheck racecheck synccheck; do
 run "${tool}_small" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "${base[@]}" --variant candidate --out "$R/${tool}_small.jsonl" --ms 1 5 8 --shapes 256,2048 \
  --configs A B C --patterns mixed zero_a zero_w --policies no_flush --repeats 2
 run "${tool}_main" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "${base[@]}" --variant candidate --out "$R/${tool}_main.jsonl" --ms 8 16 64 \
  --shapes 4096,4096 4096,11008 11008,4096 --configs A B C --policies no_flush --repeats 1
done
run validation-check "$E005_PYTHON" scripts/e007/check.py "$R" --stage validation --e006 "$E006_CONTROL/results/e006_20261106"
sha256sum "$R/control/libe005.so" "$R/candidate/libe005.so" scripts/e007/bench.py scripts/e005/bench.py scripts/e004/bench.py > "$R/validated-inputs.sha256"
