#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e011-release
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:03:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e011_c4_20261107"
[[ -e "$R/PREREQUISITE_PASSED" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/release-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/release-allocation.txt"
grep -q 'JobState=RUNNING' "$R/release-allocation.txt"
grep -q 'NodeList=c4' "$R/release-allocation.txt"
sha256sum -c "$R/verified-inputs.sha256" > "$R/release-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/release-library-check.log"
bench_base
run identity-release "$E005_PYTHON" scripts/e009/identity.py
run release-smoke "$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned" \
 --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
 --ms 4 --shapes 256,2048 --warmup 0 --iters 0 --out "$R/release-smoke.jsonl"
