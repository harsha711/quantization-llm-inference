#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e013-cap
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged project path}"
source "$PROJECT_ROOT/scripts/e013/env_arc.sh"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
allocation capability
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
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/handoff.sha256) > "$R/e006-handoff-check.log"
[[ -x "$E013_PYTHON" ]]
"$E013_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E004_CONTROL/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
run runtime "$E013_PYTHON" scripts/e004/check_runtime.py
clock > "$R/capability-clocks.csv"
touch "$R/CAPABILITY_PASSED"
