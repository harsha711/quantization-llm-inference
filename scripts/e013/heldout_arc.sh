#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-heldout
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
[[ -e "$R/PRIMARY_COMPLETE" ]]
grep -q '"heldout_required": true' "$R/primary-analysis.json"
allocation heldout
sha256sum -c "$R/mapped-libraries.sha256" > "$R/heldout-library-check.log"
sha256sum -c "$R/measurement-tooling.sha256" > "$R/heldout-tooling-check.log"
bench_base
run heldout-validation "${base[@]}" --out "$R/heldout-validation.jsonl" --seed-start 7 --warmup 0 --iters 0 --trace
clock > "$R/heldout1-clocks-before.csv"
run heldout1 "${base[@]}" --out "$R/heldout1.jsonl" --seed-start 7
clock > "$R/heldout1-clocks-after.csv"
clock > "$R/heldout2-clocks-before.csv"
run heldout2 "${base[@]}" --out "$R/heldout2.jsonl" --seed-start 7 --shapes 11008,4096 4096,11008 --series 2
clock > "$R/heldout2-clocks-after.csv"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/heldout-library-final-check.log"
run complete-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106"
