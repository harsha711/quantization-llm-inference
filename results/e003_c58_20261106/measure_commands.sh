set -euo pipefail
cd /home/hpuvvad/csc548-w4a16-e003-20261106
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR=/home/hpuvvad/csc548-w4a16-e003-20261106/results/e003_c58_20261106
export SOURCE_COMMIT=25b2291
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/allocation_before_measurement.txt"
prun bash scripts/run_inblock.sh > "$RUN_DIR/timings.log" 2>&1
prun bash scripts/ncu_inblock.sh > "$RUN_DIR/profile.log" 2>&1
tail -20 "$RUN_DIR/timings.log"
# Resource inspection was executed separately after the timing/profile series.
cuobjdump --dump-resource-usage build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_inblock_rows.cu.o > "$RUN_DIR/resource_usage.txt"
