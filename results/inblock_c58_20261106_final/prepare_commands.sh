set -euo pipefail
cd /home/hpuvvad/csc548-w4a16-inblock-20261106
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR=/home/hpuvvad/csc548-w4a16-inblock-20261106/results/inblock_c58_20261106_final
mkdir -p "$RUN_DIR"
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/allocation.txt"
bash scripts/build.sh > "$RUN_DIR/build.log" 2>&1
cuobjdump --dump-sass results/inblock_c58_20261106/original-naive.o > "$RUN_DIR/naive_original.sass"
cuobjdump --dump-sass build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_naive.cu.o > "$RUN_DIR/naive_specialized.sass"
prun bash scripts/check_inblock.sh > "$RUN_DIR/check.log" 2>&1
ln results/inblock_c58_20261106/metric_names.txt "$RUN_DIR/metric_names.txt"
tail -5 "$RUN_DIR/check.log"
