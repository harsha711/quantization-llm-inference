#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-validate
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:35:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
[[ -e "$R/DIAGNOSTIC_CAPTURED" && -e "$R/DIAGNOSTIC_INSPECTED" ]]
allocation validation
export TORCH_LOGS=graph_breaks,recompiles,cudagraphs
for context in 128 2048; do
 for variant in A B C; do
  run "validation_${variant}_${context}" "$E014_PYTHON" scripts/e014/validate.py "$R" --variant "$variant" --context "$context"
 done
done
clock > "$R/validation-clocks.csv"
touch "$R/VALIDATION_CAPTURED"
