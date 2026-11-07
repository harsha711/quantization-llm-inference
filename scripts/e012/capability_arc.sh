#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e012-cap
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
set -euo pipefail
: "${PROJECT_ROOT:?Export the absolute staged project path}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e011/profile.py" ]]
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
export PYTHONDONTWRITEBYTECODE=1
R="$PROJECT_ROOT/results/e012_c4_20261107"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/capability-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/capability-allocation.txt"
grep -q 'JobState=RUNNING' "$R/capability-allocation.txt"
grep -q 'NodeList=c4' "$R/capability-allocation.txt"
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
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
[[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) == '12.0' ]]
(cd /home/hpuvvad/csc548-w4a16-e011-c4-20261107; sha256sum -c results/e011_c4_20261107/handoff.sha256) > "$R/e011-handoff-check.log"
ncu --help > "$R/ncu-help.txt"
run query-source ncu --query-metrics --query-metrics-collection source
run query-warpsampling ncu --query-metrics --query-metrics-collection warpsampling
old=/home/hpuvvad/csc548-w4a16-e011-c4-20261107/results/e011_c4_20261107
for m in 4 128; do
 run "source_M$m" ncu --import "$old/ncu/marlin_M${m}_K4096_N11008.ncu-repz" \
  --page source --print-source cuda,sass --csv --metrics gpu__time_duration.sum
 done
run report-interface python3 scripts/e012/inspect_report.py "$(command -v ncu)" "$old/ncu/marlin_M128_K4096_N11008.ncu-repz"
sha256sum "$E005_LOCAL" "$E005_BRIDGE" scripts/e011/{profile.py,plan.py} scripts/e012/{capability_arc.sh,inspect_report.py} > "$R/capability-inputs.sha256"
touch "$R/CAPABILITY_INSPECTED"
