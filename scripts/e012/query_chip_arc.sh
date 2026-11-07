#!/usr/bin/env bash
#SBATCH --job-name=e012-chip-query
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:03:00
set -euo pipefail
: "${PROJECT_ROOT:?Export absolute project path}"
export PATH=/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=1 OMP_PROC_BIND=true PYTHONDONTWRITEBYTECODE=1
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
R="$PROJECT_ROOT/results/e012_source_c54_20261107"
[[ -e "$R/SOURCE_INSPECTED" && ! -e "$R/gb206-source-query.log" ]]
scontrol show job "$SLURM_JOB_ID" > "$R/chip-query-allocation.txt"
grep -q 'UserId=hpuvvad' "$R/chip-query-allocation.txt"
grep -q 'JobState=RUNNING' "$R/chip-query-allocation.txt"
grep -q 'NodeList=c54' "$R/chip-query-allocation.txt"
prun ncu --list-chips > "$R/supported-chips.txt" 2>&1
grep -q gb206 "$R/supported-chips.txt"
for collection in source warpsampling stats; do
 prun timeout 60 ncu --query-metrics --chips gb206 --query-metrics-collection "$collection" > "$R/gb206-$collection-query.log" 2>&1
done
touch "$R/CHIP_QUERIES_COMPLETE"
