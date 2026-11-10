#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e014-setup
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute staged root}"
source "$PROJECT_ROOT/scripts/e014/env_arc.sh"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
allocation setup
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
 nsys --version
 nvidia-smi -q
 nvidia-smi topo -m
 findmnt -T "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
[[ ! -e /tmp/e014-hpuvvad-20261108 ]]
run venv python3.11 -m venv /tmp/e014-hpuvvad-20261108
run install "$E014_PYTHON" -m pip install -r scripts/e014/requirements.txt
"$E014_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
run fetch "$E014_PYTHON" scripts/e014/fetch.py "$R"
clock > "$R/setup-clocks.csv"
touch "$R/SETUP_PASSED"
