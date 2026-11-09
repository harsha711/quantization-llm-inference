#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e016-setup-resume
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}"
cd "$PROJECT_ROOT"
r=results/e016_20261109
cp /tmp/e016-setup-331919.log "$r/failed-bootstrap-331919.log"
cp "$0" "$r/setup-resume-spooled.sh"
find scripts/e014 scripts/e015 scripts/e016 src include tests -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$r/setup-source-fixed.sha256"
MODE=setup bash scripts/e016/stage_arc.sh
