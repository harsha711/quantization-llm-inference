#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e016
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
set -euo pipefail
: "${PROJECT_ROOT:?Absolute root}" "${MODE:?setup, validation, primary, heldout}"
source "$PROJECT_ROOT/scripts/e016/env_arc.sh"
exec > "$R/$MODE-driver.log" 2>&1
trap 'rc=$?; echo "$rc" > "$R/$MODE-driver.exit"; touch "$R/${MODE^^}_FINISHED"; for ((i=0;i<300;i++)); do [[ -e "$R/${MODE^^}_RETRIEVED" ]] && break; sleep 2; done; exit "$rc"' EXIT
allocation "$MODE"
if [[ "$MODE" == setup ]]; then
 prun /bin/hostname > "$R/launcher.log" 2>&1
 [[ $(grep -c "^$SLURM_JOB_NODELIST$" "$R/launcher.log") == 1 ]]
 {
  date --iso-8601=seconds
  lscpu; lscpu -e; cat /proc/cpuinfo; numactl --hardware; cat /proc/self/status
  gcc --version; mpirun --version; nvcc --version; compute-sanitizer --version
  nvidia-smi -q; nvidia-smi topo -m
  env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
  findmnt -T "$PROJECT_ROOT"; df -h "$PROJECT_ROOT"
 } > "$R/metadata.txt" 2>&1
 [[ -x "$E016_PYTHON" ]]
 "$E016_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
 diff -u scripts/e014/requirements.lock "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
 [[ $(df --output=avail -B1 "$PROJECT_ROOT" | tail -1) -gt 4294967296 ]]
 dd if=/dev/zero of="$R/storage-probe.bin" bs=1048576 count=1
 sha256sum "$R/storage-probe.bin" > "$R/storage-probe.sha256"
 run runtime "$E016_PYTHON" scripts/e016/runtime.py "$R"
 for variant in G H; do
  run "smoke_${variant}_128" "$E016_PYTHON" scripts/e016/run.py "$R" --stage smoke --variant "$variant" --context 128
 done
elif [[ "$MODE" == validation ]]; then
 [[ -e "$R/SMOKE_INSPECTED" ]]
 run project-build bash scripts/build.sh
 run project-ctest ctest --test-dir build --output-on-failure
 run project-gpu-check bash scripts/check_inblock.sh
 sha256sum scripts/e014/{model.py,fetch.py,prompts.json,validate.py,timeline.py} scripts/e015/{runner.py,run.py,checks.py} scripts/e016/*.* > "$R/validation-tooling.sha256"
 for context in 128 2048; do
  for variant in G H; do
   run "validation_${variant}_${context}" "$E016_PYTHON" scripts/e016/run.py "$R" --stage validation --variant "$variant" --context "$context"
  done
 done
 run validation-analysis "$E016_PYTHON" scripts/e016/analyze.py "$R" --stage validation
elif [[ "$MODE" == primary || "$MODE" == heldout ]]; then
 [[ -e "$R/VALIDATION_INSPECTED" ]]
 if [[ "$MODE" == heldout ]]; then
  [[ -e "$R/PRIMARY_INSPECTED" ]]
  grep -q '"heldout_required": true' "$R/primary-analysis.json"
 fi
 sha256sum -c "$R/validation-tooling.sha256" > "$R/$MODE-tooling-check.log"
 "$E016_PYTHON" -m pip freeze > "$R/$MODE-pip-freeze.txt"
 diff -u "$R/pip-freeze.txt" "$R/$MODE-pip-freeze.txt" > "$R/$MODE-dependency-diff.txt"
 cp "$R/validation-tooling.sha256" "$R/$MODE-tooling.sha256"
 for series in 1 2; do
  contexts=(128 2048); variants=(G H)
  if [[ "$series" == 2 ]]; then contexts=(2048 128); variants=(H G); fi
  for context in "${contexts[@]}"; do
   for variant in "${variants[@]}"; do
    label="${MODE}_s${series}_${variant}_${context}"
    clock > "$R/$label-clocks-before.csv"
    run "$label" "$E016_PYTHON" scripts/e016/run.py "$R" --stage "$MODE" --variant "$variant" --context "$context" --series "$series"
    clock > "$R/$label-clocks-after.csv"
   done
  done
 done
 run "$MODE-analysis" "$E016_PYTHON" scripts/e016/analyze.py "$R" --stage "$MODE"
else
 exit 2
fi
clock > "$R/$MODE-clocks.csv"
touch "$R/${MODE^^}_CAPTURED"
