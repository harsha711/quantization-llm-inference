#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-heldout
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
[[ -e "$R/PRIMARY_COMPLETE" && -e "$R/PRIMARY_INSPECTED" ]]
grep -q '"heldout_required": true' "$R/primary-analysis.json"
allocation heldout
export TORCH_LOGS=graph_breaks,recompiles,cudagraphs
sha256sum -c "$R/measurement-tooling.sha256" > "$R/heldout-tooling-check.log"
"$E014_PYTHON" -m pip freeze > "$R/heldout-pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$R/heldout-pip-freeze.txt" > "$R/heldout-dependency-diff.txt"
for series in 1 2; do
 contexts=(128 2048)
 variants=(A B C)
 if [[ "$series" == 2 ]]; then contexts=(2048 128); variants=(C B A); fi
 for context in "${contexts[@]}"; do
  for variant in "${variants[@]}"; do
   label="heldout_s${series}_${variant}_${context}"
   clock > "$R/$label-clocks-before.csv"
   run "$label" "$E014_PYTHON" scripts/e014/measure.py "$R" --variant "$variant" --context "$context" --series "$series" --prompt heldout
   clock > "$R/$label-clocks-after.csv"
  done
 done
done
run final-analysis "$E014_PYTHON" scripts/e014/analyze.py "$R"
