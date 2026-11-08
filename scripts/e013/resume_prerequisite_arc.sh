#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-pre-resume
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
allocation prerequisite-resume
[[ $(cat "$R/smoke.exit") == 1 ]]
grep -q 'UnboundLocalError' "$R/smoke.log"
mkdir "$R/smoke-attempt1"
mv "$R"/smoke.log "$R"/smoke.exit "$R"/smoke.jsonl "$R"/smoke_*.trace.json "$R/smoke-attempt1/"
run runtime-resume "$E013_PYTHON" scripts/e004/check_runtime.py
run build-check-resume "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage build
bench_base
run smoke "${base[@]}" --out "$R/smoke.jsonl" --shapes 256,2048 --matrices 1 \
 --patterns mixed zero_a zero_w --warmup 0 --iters 0 --trace
run prerequisite-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage prerequisite
"$E013_PYTHON" scripts/e009/libraries.py "$R" > "$R/library-identity.log"
touch "$R/PREREQUISITE_PASSED"
