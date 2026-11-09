#!/usr/bin/env bash
# Run from an allocated absolute project root after the dependency/smoke gate.
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}"
cd "$PROJECT_ROOT"
R="$PROJECT_ROOT/results/e016_20261109"
wait_for() {
 local marker=$1
 for ((i=0;i<600;i++)); do
  [[ -e "$marker" ]] && return 0
  sleep 2
 done
 return 124
}
source scripts/e016/env_arc.sh
allocation continuation
[[ -e "$R/UPDATED_SMOKE_INSPECTED" ]]
sha256sum -c "$R/updated-prerequisite-tooling.sha256" > "$R/continuation-source-check.log"
MODE=validation bash scripts/e016/stage_arc.sh
[[ -e "$R/VALIDATION_INSPECTED" ]]
MODE=primary bash scripts/e016/stage_arc.sh
[[ -e "$R/PRIMARY_INSPECTED" ]]
if grep -q '"heldout_required": true' "$R/primary-analysis.json"; then
 MODE=heldout bash scripts/e016/stage_arc.sh
fi
