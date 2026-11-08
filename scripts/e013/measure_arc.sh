#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-timing
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
[[ -e "$R/VALIDATION_PASSED" ]]
allocation measurement
sha256sum -c "$R/mapped-libraries.sha256" > "$R/measurement-library-check.log"
bench_base
(cd "$R"; sha256sum libe005.so matched-source.tar.gz e013-source.json sass.txt.gz) > "$R/measurement-inputs.sha256"
sha256sum scripts/e013/*.* scripts/e005/{bench.py,local.cu} scripts/e004/bench.py scripts/e008/bench.py scripts/e010/bench.py > "$R/measurement-tooling.sha256"
clock > "$R/series1-clocks-before.csv"
run series1 "${base[@]}" --out "$R/series1.jsonl"
clock > "$R/series1-clocks-after.csv"
clock > "$R/series2-clocks-before.csv"
run series2 "${base[@]}" --out "$R/series2.jsonl" --shapes 11008,4096 4096,11008 --series 2
clock > "$R/series2-clocks-after.csv"
run primary-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage primary
sha256sum -c "$R/mapped-libraries.sha256" > "$R/measurement-library-final-check.log"
touch "$R/PRIMARY_COMPLETE"
