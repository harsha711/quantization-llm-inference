#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e010/env_arc.sh"
[[ $(cat "$R/validation.exit") == 0 && $(cat "$R/memcheck.exit") == 0 ]]
[[ $(cat "$R/racecheck-composed-attempt1.exit") == 143 ]]
[[ ! -e "$R/VALIDATION_PASSED" ]]
cycle_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/resume-input-check.log"
representative_racecheck
run synccheck compute-sanitizer --tool synccheck --error-exitcode 99 \
 "${base[@]}" --out "$R/synccheck.jsonl" --implementations marlin --warmup 0 --iters 0
"$E005_PYTHON" scripts/e010/analyze.py "$R" --stage validation > "$R/validation-check.log"
touch "$R/VALIDATION_PASSED"
