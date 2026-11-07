#!/usr/bin/env bash
#SBATCH --job-name=e012-source-api
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute project path}"
export PATH=/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=1 OMP_PROC_BIND=true PYTHONDONTWRITEBYTECODE=1
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$PROJECT_ROOT/results/e012_source_c54_20261107"
[[ -s "$R/source_M128.csv" && -e "$R/report-interface.log" && ! -e "$R/SOURCE_INSPECTED" ]]
[[ ! -e "$R/report-interface-attempt2.log" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/resume-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/resume-allocation.txt"
grep -q 'JobState=RUNNING' "$R/resume-allocation.txt"
grep -q 'NodeList=c54' "$R/resume-allocation.txt"
ncu --version > "$R/resume-ncu-version.txt"
grep -q '2026.3.1.0' "$R/resume-ncu-version.txt"
ls -ld /opt/nvidia/nsight-compute/2026.3.1/extras/python > "$R/report-package.txt"
old=/home/hpuvvad/csc548-w4a16-e011-c4-20261107/results/e011_c4_20261107
prun timeout 60 python3 "$PROJECT_ROOT/scripts/e012/inspect_report.py" "$(command -v ncu)" \
 "$old/ncu/marlin_M128_K4096_N11008.ncu-repz" > "$R/report-interface-attempt2.log" 2>&1
sha256sum "$PROJECT_ROOT/scripts/e012/inspect_report.py" "$PROJECT_ROOT/scripts/e012/resume_source_inspection_arc.sh" \
 "$old/ncu/marlin_M4_K4096_N11008.ncu-repz" "$old/ncu/marlin_M128_K4096_N11008.ncu-repz" > "$R/inputs.sha256"
touch "$R/SOURCE_INSPECTED"
