#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e015-setup
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e015/env_arc.sh"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
allocation setup
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c7$' "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 lscpu; lscpu -e; cat /proc/cpuinfo; numactl --hardware; cat /proc/self/status
 gcc --version; mpirun --version; nvcc --version; compute-sanitizer --version
 nvidia-smi -q; nvidia-smi topo -m; findmnt -T "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
[[ ! -e /tmp/e015-hpuvvad-20261109 ]]
run venv python3.11 -m venv /tmp/e015-hpuvvad-20261109
run install "$E015_PYTHON" -m pip install -r scripts/e014/requirements.lock
"$E015_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u scripts/e014/requirements.lock "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
cp "$PROJECT_ROOT/results/e014_20261108/model.json" "$R/model.json"
for context in 128 2048; do
 cp "$PROJECT_ROOT/results/e014_20261108/validation_B_${context}.json" "$R/e014-B-$context.json"
done
run local-check "$E015_PYTHON" scripts/e015/check_local.py
for variant in E G; do
 run "smoke_${variant}_128" "$E015_PYTHON" scripts/e015/run.py "$R" --stage smoke --variant "$variant" --context 128
done
clock > "$R/setup-clocks.csv"
touch "$R/SMOKE_CAPTURED"
