#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e015-validate
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e015/env_arc.sh"
[[ -e "$R/SMOKE_INSPECTED" ]]
allocation validation
sha256sum scripts/e014/{model.py,fetch.py,prompts.json,validate.py,timeline.py} scripts/e015/*.* > "$R/validation-tooling.sha256"
run project-build bash scripts/build.sh
run project-ctest ctest --test-dir build --output-on-failure
run project-gpu-check bash scripts/check_inblock.sh
for context in 128 2048; do
 for variant in E G; do
  run "validation_${variant}_${context}" "$E015_PYTHON" scripts/e015/run.py "$R" --stage validation --variant "$variant" --context "$context"
 done
done
run validation-analysis "$E015_PYTHON" scripts/e015/analyze.py "$R" --stage validation
clock > "$R/validation-clocks.csv"
touch "$R/VALIDATION_CAPTURED"
