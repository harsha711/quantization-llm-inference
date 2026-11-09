#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e015-timing
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e015/env_arc.sh"
: "${PHASE:?primary or heldout}"
[[ "$PHASE" == primary || "$PHASE" == heldout ]]
[[ -e "$R/VALIDATION_INSPECTED" ]]
if [[ "$PHASE" == heldout ]]; then
 [[ -e "$R/PRIMARY_INSPECTED" ]]
 grep -q '"heldout_required": true' "$R/primary-analysis.json"
fi
allocation "$PHASE"
"$E015_PYTHON" -m pip freeze > "$R/$PHASE-pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$R/$PHASE-pip-freeze.txt" > "$R/$PHASE-dependency-diff.txt"
sha256sum scripts/e014/{model.py,fetch.py,prompts.json,validate.py,timeline.py} scripts/e015/*.* > "$R/$PHASE-tooling.sha256"
for series in 1 2; do
 contexts=(128 2048); variants=(E G)
 if [[ "$series" == 2 ]]; then contexts=(2048 128); variants=(G E); fi
 for context in "${contexts[@]}"; do
  for variant in "${variants[@]}"; do
   label="${PHASE}_s${series}_${variant}_${context}"
   clock > "$R/$label-clocks-before.csv"
   run "$label" "$E015_PYTHON" scripts/e015/run.py "$R" --stage "$PHASE" --variant "$variant" --context "$context" --series "$series"
   clock > "$R/$label-clocks-after.csv"
  done
 done
done
run "$PHASE-analysis" "$E015_PYTHON" scripts/e015/analyze.py "$R" --stage "$PHASE"
