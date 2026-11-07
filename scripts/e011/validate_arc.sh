#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e011-check
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e011_c4_20261107"
[[ -e "$R/PREREQUISITE_PASSED" && ! -e "$R/TARGETS_VALIDATED" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/validation-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/validation-allocation.txt"
grep -q 'JobState=RUNNING' "$R/validation-allocation.txt"
grep -q 'NodeList=c4' "$R/validation-allocation.txt"
sha256sum -c "$R/verified-inputs.sha256" > "$R/validation-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/validation-library-check.log"
bench_base
run identity-validation "$E005_PYTHON" scripts/e009/identity.py
prun ncu --query-metrics --query-metrics-collection launch > "$R/launch-metrics.txt" 2>&1
cp /home/hpuvvad/csc548-w4a16-e010-c4-20261107/results/e010_c4_20261107/validation.jsonl "$R/e010-validation-reference.jsonl"
mkdir "$R/ncu"
run calibration ncu --replay-mode application --app-replay-mode strict --app-replay-match grid \
 --profile-from-start off --graph-profiling node --cache-control none --clock-control none \
 --pipeline-boost-state dynamic --launch-skip 399 --launch-count 1 \
 --metrics gpu__time_duration.sum,launch__grid_size --export "$R/ncu/calibration" \
 "$E005_PYTHON" scripts/e011/calibrate.py
ncu --import "$R/ncu/calibration.ncu-repz" --page raw --csv --print-units base > "$R/ncu/calibration.csv"
for shape in 4096,11008 11008,4096; do
 IFS=, read -r k n <<< "$shape"
 for m in 4 128; do
  for implementation in marlin cublas_dq; do
   tag="trace_${implementation}_M${m}_K${k}_N${n}"
   run "$tag" "$E005_PYTHON" scripts/e011/profile.py --source "$E005_CONTROL/external/pinned" \
    --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
    --m "$m" --k "$k" --n "$n" --implementation "$implementation" --trace --out "$R/$tag.json"
  done
 done
done
"$E005_PYTHON" scripts/e011/analyze.py "$R" --stage validation > "$R/validation-analysis.log"
touch "$R/TARGETS_VALIDATED"
