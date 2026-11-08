#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-pre
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
[[ -e "$R/CAPABILITY_PASSED" ]]
allocation prerequisite
run prepare-source "$E013_PYTHON" scripts/e013/prepare_source.py \
 results/e006_20261106/matched-source.tar.gz "$E013_SOURCE"
run build bash scripts/e013/build.sh
tar -czf "$R/matched-source.tar.gz" -C "$E013_SOURCE" .
run build-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage build
bench_base
run smoke "${base[@]}" --out "$R/smoke.jsonl" --shapes 256,2048 --matrices 1 \
 --patterns mixed zero_a zero_w --warmup 0 --iters 0 --trace
run prerequisite-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage prerequisite
"$E013_PYTHON" scripts/e009/libraries.py "$R" > "$R/library-identity.log"
touch "$R/PREREQUISITE_PASSED"
