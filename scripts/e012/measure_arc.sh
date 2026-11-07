#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e012-pc
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e012_c4_20261107"
[[ -e "$R/VALIDATION_PASSED" && -e "$R/RELEASED_AFTER_INSPECTION" && ! -e "$R/MEASUREMENTS_COMPLETE" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/measurement-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/measurement-allocation.txt"
grep -q 'JobState=RUNNING' "$R/measurement-allocation.txt"
grep -q 'NodeList=c4' "$R/measurement-allocation.txt"
[[ $(nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader) == 'GPU-f2e08184-f913-0d97-104d-e01716a3c597, 610.57.04' ]]
bench_base
sha256sum -c "$R/verified-inputs.sha256" > "$R/measurement-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/measurement-library-check.log"
run identity-measurement "$E005_PYTHON" scripts/e009/identity.py
sha256sum scripts/e012/{measure_arc.sh,export_pc.py,regions.py,instruction-metrics.txt,sampling-metrics.txt} \
 scripts/e011/{profile.py,plan.py} scripts/e010/bench.py scripts/e008/bench.py scripts/e005/bench.py \
 scripts/e004/{bench.py,bridge.cu} src/cuda/{dequant.cu,cublas_gemm.cu} > "$R/measurement-tooling.sha256"
cp scripts/e012/instruction-metrics.txt scripts/e012/sampling-metrics.txt "$R/"
mkdir "$R/ncu"
profile_case() {
 local kind=$1 series=$2 m=$3 k=$4 n=$5
 local label="${kind}_s${series}_M${m}_K${k}_N${n}"
 local extra=() metrics_file
 if [[ "$kind" == instruction ]]; then
  metrics_file=scripts/e012/instruction-metrics.txt
 else
  metrics_file=scripts/e012/sampling-metrics.txt
  extra=(--warp-sampling-interval 0 --warp-sampling-max-passes 1 --warp-sampling-buffer-size 33554432)
 fi
 local metrics
 metrics=$(paste -sd, "$metrics_file")
 clock > "$R/$label-clocks-before.csv"
 run "$label" ncu --replay-mode application --app-replay-mode strict --app-replay-match grid \
  --profile-from-start off --graph-profiling node --cache-control none --clock-control none \
  --pipeline-boost-state dynamic --disable-extra-suffixes --launch-skip 399 --launch-count 1 \
  "${extra[@]}" --metrics "$metrics" --export "$R/ncu/$label" \
  "$E005_PYTHON" scripts/e011/profile.py --source "$E005_CONTROL/external/pinned" \
  --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
  --m "$m" --k "$k" --n "$n" --implementation marlin
 clock > "$R/$label-clocks-after.csv"
 ncu --import "$R/ncu/$label.ncu-repz" --page raw --csv --print-units base > "$R/ncu/$label.raw.csv"
 ncu --import "$R/ncu/$label.ncu-repz" --page source --print-source cuda,sass --csv \
  --print-units base --metrics "$metrics" > "$R/ncu/$label.source.csv"
 run "export_$label" "$E005_PYTHON" scripts/e012/export_pc.py --ncu "$(command -v ncu)" \
  --report "$R/ncu/$label.ncu-repz" --source-csv "$R/ncu/$label.source.csv" --metrics "$metrics_file" \
  --e006 "$E006_CONTROL/results/e006_20261106" --out "$R/ncu/$label.pc.json.gz"
}
for specification in 4,4096,11008 128,4096,11008 4,11008,4096 128,11008,4096; do
 IFS=, read -r m k n <<< "$specification"
 profile_case instruction 1 "$m" "$k" "$n"
done
for specification in 4,4096,11008 128,4096,11008 4,11008,4096 128,11008,4096; do
 IFS=, read -r m k n <<< "$specification"
 profile_case sampling 1 "$m" "$k" "$n"
done
for specification in 128,11008,4096 4,11008,4096 128,4096,11008 4,4096,11008; do
 IFS=, read -r m k n <<< "$specification"
 profile_case sampling 2 "$m" "$k" "$n"
done
run identity-final "$E005_PYTHON" scripts/e009/identity.py
sha256sum -c "$R/verified-inputs.sha256" > "$R/final-input-check.log"
sha256sum -c "$R/mapped-libraries.sha256" > "$R/final-library-check.log"
sha256sum -c "$R/measurement-tooling.sha256" > "$R/final-tooling-check.log"
touch "$R/COLLECTION_COMPLETE"
