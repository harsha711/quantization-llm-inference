#!/usr/bin/env bash
#SBATCH --job-name=e012-analysis
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute project path}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=1 OMP_PROC_BIND=true PYTHONDONTWRITEBYTECODE=1
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$PROJECT_ROOT/results/e012_c4_20261107"
[[ -e "$R/COLLECTION_COMPLETE" && ! -e "$R/arc-analysis.log" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/analysis-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/analysis-allocation.txt"
grep -q 'JobState=RUNNING' "$R/analysis-allocation.txt"
grep -q 'NodeList=c54' "$R/analysis-allocation.txt"
prun /bin/hostname > "$R/analysis-launcher.log" 2>&1
[[ $(grep -c '^c54$' "$R/analysis-launcher.log") == 1 ]]
sha256sum "$R"/ncu/* > "$R/raw-profiles.sha256"
old=/home/hpuvvad/csc548-w4a16-e011-c4-20261107/results/e011_c4_20261107
e006=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106
prun python3 scripts/e012/analyze.py "$R" "$old" "$e006" > "$R/arc-analysis.log" 2>&1
prun python3 scripts/e012/check_analysis.py "$R" "$old" "$e006" > "$R/arc-negative-analysis.log" 2>&1
sha256sum -c "$R/raw-profiles.sha256" > "$R/analysis-raw-check.log"
sha256sum scripts/e012/{analyze.py,check_analysis.py,check_validation.py} > "$R/analysis-tooling.sha256"
