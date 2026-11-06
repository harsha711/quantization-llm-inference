set -euo pipefail
cd /home/hpuvvad/csc548-w4a16-e003-20261106
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR=/home/hpuvvad/csc548-w4a16-e003-20261106/results/e003_c58_20261106
scontrol show job "$SLURM_JOB_ID" > "$RUN_DIR/allocation.txt"
prun /bin/hostname > "$RUN_DIR/launcher.log" 2>&1
prun bash scripts/build.sh > "$RUN_DIR/build.log" 2>&1
cuobjdump --dump-sass /home/hpuvvad/csc548-w4a16-e002-20261106/build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_inblock_specialized.cu.o > "$RUN_DIR/e002_saved.sass"
cuobjdump --dump-sass /home/hpuvvad/csc548-w4a16-e002-20261106/build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_naive.cu.o > "$RUN_DIR/naive_saved.sass"
cuobjdump --dump-sass build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_inblock_specialized.cu.o > "$RUN_DIR/e002_rebuilt.sass"
cuobjdump --dump-sass build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_naive.cu.o > "$RUN_DIR/naive_rebuilt.sass"
cuobjdump --dump-sass build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_inblock_rows.cu.o > "$RUN_DIR/e003.sass"
prun ./build/w4a16_bench --M 5 --N 33 --K 137 --group 128 --seed 1 --warmup 1 --iters 2 > "$RUN_DIR/smoke.log" 2>&1
prun bash scripts/check_inblock.sh > "$RUN_DIR/check.log" 2>&1
tail -7 "$RUN_DIR/check.log"
