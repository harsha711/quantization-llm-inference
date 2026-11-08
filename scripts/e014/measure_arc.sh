#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-primary
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
[[ -e "$R/VALIDATION_INSPECTED" ]]
allocation primary
export TORCH_LOGS=graph_breaks,recompiles,cudagraphs
"$E014_PYTHON" -m pip freeze > "$R/measurement-pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$R/measurement-pip-freeze.txt" > "$R/measurement-dependency-diff.txt"
sha256sum scripts/e014/*.* > "$R/measurement-tooling.sha256"
for series in 1 2; do
 contexts=(128 2048)
 variants=(A B C)
 if [[ "$series" == 2 ]]; then contexts=(2048 128); variants=(C B A); fi
 for context in "${contexts[@]}"; do
  for variant in "${variants[@]}"; do
   label="primary_s${series}_${variant}_${context}"
   clock > "$R/$label-clocks-before.csv"
   run "$label" "$E014_PYTHON" scripts/e014/measure.py "$R" --variant "$variant" --context "$context" --series "$series" --prompt primary
   clock > "$R/$label-clocks-after.csv"
  done
 done
done
run primary-analysis "$E014_PYTHON" scripts/e014/analyze.py "$R" --stage primary
touch "$R/PRIMARY_COMPLETE"
