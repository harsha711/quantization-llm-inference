#!/usr/bin/env bash
# Submit from home storage with an exported absolute PROJECT_ROOT.
#SBATCH --job-name=w4a16-e011-pre
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
set -euo pipefail
: "${PROJECT_ROOT:?Export the absolute staged project path}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e010/bench.py" ]]
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e011_c4_20261107"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
grep -q 'UserId=hpuvvad' "$R/allocation.txt"
grep -q 'JobState=RUNNING' "$R/allocation.txt"
grep -q 'NodeList=c4' "$R/allocation.txt"
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c4$' "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 lscpu
 lscpu -e
 cat /proc/cpuinfo
 numactl --hardware
 cat /proc/self/status
 gcc --version
 mpirun --version
 nvcc --version
 compute-sanitizer --version
 ncu --version
 nvidia-smi -q
 nvidia-smi topo -m
 findmnt -T "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|LD_LIBRARY_PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
ncu --help > "$R/ncu-help.txt"
prun ncu --query-metrics --query-metrics-mode all > "$R/supported-metrics.txt" 2>&1
(cd /home/hpuvvad/csc548-w4a16-e010-c4-20261107; sha256sum -c results/e010_c4_20261107/handoff.sha256) > "$R/e010-handoff-check.log"
bench_base
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E004_CONTROL/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
run identity "$E005_PYTHON" scripts/e009/identity.py
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run smoke "$E005_PYTHON" scripts/e010/bench.py --source "$E005_CONTROL/external/pinned" \
 --local "$E005_LOCAL" --wheel "$E005_WHEEL" --bridge "$E005_BRIDGE" \
 --ms 4 128 --shapes 256,2048 --warmup 0 --iters 0 --out "$R/smoke.jsonl"
run library-identity "$E005_PYTHON" scripts/e009/libraries.py "$R"
sha256sum "$E005_LOCAL" "$E005_BRIDGE" "$E005_WHEEL" > "$R/verified-inputs.sha256"
touch "$R/PREREQUISITE_PASSED"
