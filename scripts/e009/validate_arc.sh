#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e009/env_arc.sh"
[[ -e "$R/SMOKE_PASSED" ]]
cycle_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/validation-input-check.log"
run project-build bash scripts/build.sh
run project-ctest ctest --test-dir build --output-on-failure
run project-gpu bash scripts/check_inblock.sh
run validation "${base[@]}" --out "$R/validation.jsonl" --warmup 0 --iters 0
for tool in memcheck racecheck synccheck; do
 filters=()
 if [[ "$tool" == racecheck ]]; then
  filters=(--kernel-name kns=e005_marlin --kernel-name kns=dequant_w4a16_kernel)
 fi
 run "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 "${filters[@]}" \
  "${base[@]}" --out "$R/$tool.jsonl" --counts 4 --warmup 0 --iters 0
done
touch "$R/VALIDATION_PASSED"
