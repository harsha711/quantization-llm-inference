#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e011-profile
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e011_c4_20261107"
[[ -e "$R/TARGETS_VALIDATED" && ! -e "$R/MEASUREMENTS_COMPLETE" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/measurement-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/measurement-allocation.txt"
grep -q 'JobState=RUNNING' "$R/measurement-allocation.txt"
grep -q 'NodeList=c4' "$R/measurement-allocation.txt"
bench_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/measurement-library-check.log"
sha256sum scripts/e011/{profile.py,plan.py,calibrate.py,metrics.txt,measure_arc.sh} scripts/e010/bench.py scripts/e008/bench.py scripts/e005/bench.py \
 scripts/e004/{bench.py,bridge.cu} src/cuda/{dequant.cu,cublas_gemm.cu} > "$R/measurement-tooling.sha256"
run identity-measurement "$E005_PYTHON" scripts/e009/identity.py
base=("$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned"
 --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE")
clock > "$R/series1-clocks-before.csv"
run series1 "${base[@]}" --ms 4 128 --out "$R/series1.jsonl"
clock > "$R/series1-clocks-after.csv"
clock > "$R/series2-clocks-before.csv"
run series2 "${base[@]}" --ms 128 4 --shapes 11008,4096 4096,11008 --series 2 --out "$R/series2.jsonl"
clock > "$R/series2-clocks-after.csv"
cp scripts/e011/metrics.txt "$R/profile-metrics.txt"
metrics=$(paste -sd, scripts/e011/metrics.txt)
for shape in 4096,11008 11008,4096; do
 IFS=, read -r k n <<< "$shape"
 for m in 4 128; do
  for implementation in marlin cublas_dq; do
   tag="${implementation}_M${m}_K${k}_N${n}"
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
    --m "$m" --k "$k" --n "$n" --implementation "$implementation"
   clock > "$R/profile_$tag-clocks-after.csv"
   ncu --import "$R/ncu/$tag.ncu-repz" --page raw --csv --print-units base > "$R/ncu/$tag.csv"
  done
 done
done
run identity-final "$E005_PYTHON" scripts/e009/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/final-library-check.log"
sha256sum -c "$R/measurement-tooling.sha256" > "$R/final-tooling-check.log"
(cd /home/hpuvvad/csc548-w4a16-e010-c4-20261107; sha256sum -c results/e010_c4_20261107/handoff.sha256) > "$R/final-e010-handoff-check.log"
"$E005_PYTHON" scripts/e011/analyze.py "$R" > "$R/complete-analysis.log"
"$E005_PYTHON" scripts/e011/check_analysis.py "$R" > "$R/negative-analysis.log"
touch "$R/MEASUREMENTS_COMPLETE"
