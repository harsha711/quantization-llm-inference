#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-diagnose
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
[[ -e "$R/SETUP_PASSED" ]]
allocation diagnose
run diagnose "$E014_PYTHON" scripts/e014/diagnose.py "$R"
clock > "$R/diagnose-clocks.csv"
touch "$R/DIAGNOSTIC_CAPTURED"
