#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
[[ -e "$R/SMOKE_PASSED" ]]
bench_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/validation-input-check.log"
run project-build bash scripts/build.sh
run project-ctest ctest --test-dir build --output-on-failure
run project-gpu bash scripts/check_inblock.sh
run validation "${base[@]}" --out "$R/validation.jsonl" --shapes 4096,11008 11008,4096 \
 --patterns mixed zero_a zero_w --warmup 0 --iters 0 --policies no_flush
for tool in memcheck racecheck synccheck; do
 run "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
  "${base[@]}" --out "$R/$tool.jsonl" --shapes 4096,11008 11008,4096 \
  --warmup 0 --iters 0 --policies no_flush
 done
touch "$R/VALIDATION_PASSED"
