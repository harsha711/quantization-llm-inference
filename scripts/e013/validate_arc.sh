#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-validation
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
[[ -e "$R/PREREQUISITE_PASSED" ]]
allocation validation
sha256sum -c "$R/mapped-libraries.sha256" > "$R/validation-library-check.log"
run project-build bash scripts/build.sh
run project-ctest ctest --test-dir build --output-on-failure
run project-gpu bash scripts/check_inblock.sh
bench_base
run validation "${base[@]}" --out "$R/validation.jsonl" --warmup 0 --iters 0 --trace
for tool in memcheck racecheck synccheck; do
 if [[ "$tool" == racecheck ]]; then
  run "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=e005_marlin \
   "${base[@]}" --out "$R/$tool.jsonl" --matrices 1 --warmup 0 --iters 0
 else
  run "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 \
   "${base[@]}" --out "$R/$tool.jsonl" --warmup 0 --iters 0
 fi
done
run validation-check "$E013_PYTHON" scripts/e013/analyze.py "$R" --e006 "$E006_CONTROL/results/e006_20261106" --stage validation
sha256sum -c "$R/mapped-libraries.sha256" > "$R/validation-library-final-check.log"
touch "$R/VALIDATION_PASSED"
