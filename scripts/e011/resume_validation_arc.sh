#!/usr/bin/env bash
# Only for the preserved successful calibration / failed-extension-import checkpoint.
#SBATCH --job-name=w4a16-e011-resume
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
[[ $(cat "$R/calibration.exit") == 0 && -e "$R/ncu/calibration.ncu-repz" ]]
[[ -e "$R/ncu/calibration.csv" && ! -e "$R/ncu/calibration-import-attempt1.txt" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/resume-validation-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/resume-validation-allocation.txt"
grep -q 'JobState=RUNNING' "$R/resume-validation-allocation.txt"
grep -q 'NodeList=c4' "$R/resume-validation-allocation.txt"
sha256sum -c "$R/verified-inputs.sha256" > "$R/resume-validation-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/resume-validation-library-check.log"
bench_base
run identity-resume-validation "$E005_PYTHON" scripts/e009/identity.py
mv "$R/ncu/calibration.csv" "$R/ncu/calibration-import-attempt1.txt"
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
