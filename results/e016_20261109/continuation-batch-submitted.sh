#!/usr/bin/env bash
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c58
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=w4a16-e016-complete
#SBATCH --time=00:45:00
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}"
exec bash "$PROJECT_ROOT/scripts/e016/continuation_arc.sh"
