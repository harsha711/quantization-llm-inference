#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-seal
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
[[ $(cat "$R/validation_C_2048.exit") == 1 ]]
[[ ! -e "$R/VALIDATION_INSPECTED" ]]
allocation seal
run seal "$E014_PYTHON" scripts/e014/seal.py "$R"
"$E014_PYTHON" -m pip freeze > "$R/final-pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$R/final-pip-freeze.txt" > "$R/final-dependency-diff.txt"
run stopped-analysis "$E014_PYTHON" scripts/e014/analyze.py "$R"
run negative-check "$E014_PYTHON" scripts/e014/check_analysis.py "$R"
clock > "$R/final-clocks.csv"
