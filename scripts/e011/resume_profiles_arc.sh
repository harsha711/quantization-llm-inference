#!/usr/bin/env bash
# Only for the preserved 329066 timeout: five profiles complete, final three pending.
#SBATCH --job-name=w4a16-e011-finish
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e011_c4_20261107"
[[ -e "$R/TARGETS_VALIDATED" && ! -e "$R/MEASUREMENTS_COMPLETE" ]]
grep -q TIMEOUT "$R/measurement-timeout-accounting.txt"
scontrol show job "$SLURM_JOB_ID" > "$R/resume-profile-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/resume-profile-allocation.txt"
grep -q 'JobState=RUNNING' "$R/resume-profile-allocation.txt"
grep -q 'NodeList=c4' "$R/resume-profile-allocation.txt"
bench_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/resume-profile-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/resume-profile-library-check.log"
sha256sum -c "$R/measurement-tooling.sha256" > "$R/resume-profile-tooling-check.log"
run identity-resume-profile "$E005_PYTHON" scripts/e009/identity.py
for tag in marlin_M4_K4096_N11008 cublas_dq_M4_K4096_N11008 marlin_M128_K4096_N11008 cublas_dq_M128_K4096_N11008 marlin_M4_K11008_N4096; do
 [[ $(cat "$R/profile_$tag.exit") == 0 && -e "$R/ncu/$tag.ncu-repz" && -e "$R/ncu/$tag.csv" ]]
done
tag=cublas_dq_M4_K11008_N4096
[[ -e "$R/profile_$tag.log" && ! -e "$R/profile_$tag.exit" && ! -e "$R/ncu/$tag.ncu-repz" ]]
mv "$R/profile_$tag.log" "$R/profile_$tag.attempt1.log"
mv "$R/profile_$tag-clocks-before.csv" "$R/profile_$tag.attempt1-clocks-before.csv"
metrics=$(paste -sd, scripts/e011/metrics.txt)
for specification in cublas_dq,4 marlin,128 cublas_dq,128; do
 IFS=, read -r implementation m <<< "$specification"
 tag="${implementation}_M${m}_K11008_N4096"
 read -r skip count < <("$E005_PYTHON" - "$R/trace_$tag.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
print(row['launch_skip'], row['launch_count'])
PY
)
 clock > "$R/profile_$tag-clocks-before.csv"
 run "profile_$tag" ncu --replay-mode application --app-replay-mode strict --app-replay-match grid \
  --profile-from-start off --graph-profiling node --cache-control none --clock-control none \
  --pipeline-boost-state dynamic --disable-extra-suffixes --launch-skip "$skip" --launch-count "$count" \
  --metrics "$metrics" --export "$R/ncu/$tag" \
  "$E005_PYTHON" scripts/e011/profile.py --source "$E005_CONTROL/external/pinned" \
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
  --m "$m" --k 11008 --n 4096 --implementation "$implementation"
 clock > "$R/profile_$tag-clocks-after.csv"
 ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv --print-units base > "$R/ncu/$tag.csv"
done
run identity-final "$E005_PYTHON" scripts/e009/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/final-library-check.log"
sha256sum -c "$R/measurement-tooling.sha256" > "$R/final-tooling-check.log"
(cd /home/hpuvvad/csc548-w4a16-e011-c4-20261107; sha256sum -c "$R/measurement-tooling.sha256") > "$R/final-source-check.log"
(cd /home/hpuvvad/csc548-w4a16-e010-c4-20261107; sha256sum -c results/e010_c4_20261107/handoff.sha256) > "$R/final-e010-handoff-check.log"
"$E005_PYTHON" scripts/e011/analyze.py "$R" > "$R/complete-analysis.log"
"$E005_PYTHON" scripts/e011/check_analysis.py "$R" > "$R/negative-analysis.log"
touch "$R/MEASUREMENTS_COMPLETE"
