#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e012-gate
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e012_c4_20261107"
old=/home/hpuvvad/csc548-w4a16-e011-c4-20261107/results/e011_c4_20261107
[[ $(cat "$R/query-source.exit") == 0 && $(cat "$R/query-warpsampling.exit") == 0 ]]
[[ $(cat "$R/report-interface.exit") == 1 && ! -e "$R/VALIDATION_PASSED" ]]
[[ -x "$E005_PYTHON" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/validation-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/validation-allocation.txt"
grep -q 'JobState=RUNNING' "$R/validation-allocation.txt"
grep -q 'NodeList=c4' "$R/validation-allocation.txt"
prun /bin/hostname > "$R/validation-launcher.log" 2>&1
[[ $(grep -c '^c4$' "$R/validation-launcher.log") == 1 ]]
[[ $(nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader) == 'GPU-f2e08184-f913-0d97-104d-e01716a3c597, 610.57.04' ]]
bench_base
run report-interface-resume python3 scripts/e012/inspect_report.py "$(command -v ncu)" "$old/ncu/marlin_M128_K4096_N11008.ncu-repz"
run query-stats ncu --query-metrics --query-metrics-collection stats
run identity "$E005_PYTHON" scripts/e009/identity.py
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E004_CONTROL/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run smoke "$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned" \
 --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
 --ms 4 128 --shapes 256,2048 --warmup 0 --iters 0 --out "$R/smoke.jsonl"
run library-identity "$E005_PYTHON" scripts/e009/libraries.py "$R"
sha256sum -c "$old/mapped-libraries.sha256" > "$R/reused-library-check.log"
sha256sum -c "$old/measurement-tooling.sha256" > "$R/reused-tooling-check.log"
(cd /home/hpuvvad/csc548-w4a16-e010-c4-20261107; sha256sum -c "$old/reused-e010-project-build.sha256") > "$R/retained-project-build-check.log"
run retained-project-gpu bash /home/hpuvvad/csc548-w4a16-e010-c4-20261107/scripts/check_inblock.sh
sha256sum "$E005_LOCAL" "$E005_BRIDGE" "$E005_WHEEL" > "$R/verified-inputs.sha256"
for shape in 4096,11008 11008,4096; do
 IFS=, read -r k n <<< "$shape"
 for m in 4 128; do
  tag="trace_marlin_M${m}_K${k}_N${n}"
  run "$tag" "$E005_PYTHON" scripts/e011/profile.py --source "$E005_CONTROL/external/pinned" \
   --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
   --m "$m" --k "$k" --n "$n" --implementation marlin --trace --out "$R/$tag.json"
 done
done
"$E005_PYTHON" scripts/e012/check_validation.py "$R" "$old" > "$R/validation-analysis.log"
touch "$R/VALIDATION_PASSED"
