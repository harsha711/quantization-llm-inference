#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e012-source
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute project path}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e012/inspect_report.py" ]]
export PATH=/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=1 OMP_PROC_BIND=true PYTHONDONTWRITEBYTECODE=1
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$PROJECT_ROOT/results/e012_source_c54_20261107"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
grep -q 'UserId=hpuvvad' "$R/allocation.txt"
grep -q 'JobState=RUNNING' "$R/allocation.txt"
grep -q 'NodeList=c54' "$R/allocation.txt"
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c54$' "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 lscpu
 lscpu -e
 cat /proc/cpuinfo
 numactl --hardware
 cat /proc/self/status
 gcc --version
 mpirun --version
 ncu --version
 nvidia-smi -q
 findmnt -T "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
old=/home/hpuvvad/csc548-w4a16-e011-c4-20261107/results/e011_c4_20261107
cd /home/hpuvvad/csc548-w4a16-e006-20261106
for m in 4 128; do
 prun timeout 120 ncu --import "$old/ncu/marlin_M${m}_K4096_N11008.ncu-repz" \
  --page source --print-source cuda,sass --csv --metrics gpu__time_duration.sum > "$R/source_M$m.csv" 2> "$R/source_M$m.log"
done
prun timeout 120 python3 "$PROJECT_ROOT/scripts/e012/inspect_report.py" "$(command -v ncu)" \
 "$old/ncu/marlin_M128_K4096_N11008.ncu-repz" > "$R/report-interface.log" 2>&1
sha256sum "$PROJECT_ROOT/scripts/e012/inspect_report.py" "$PROJECT_ROOT/scripts/e012/source_inspection_arc.sh" \
 "$old/ncu/marlin_M4_K4096_N11008.ncu-repz" "$old/ncu/marlin_M128_K4096_N11008.ncu-repz" > "$R/inputs.sha256"
touch "$R/SOURCE_INSPECTED"
