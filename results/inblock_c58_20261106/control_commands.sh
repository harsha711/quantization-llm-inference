set -euo pipefail
cd /home/hpuvvad/csc548-w4a16-inblock-20261106
export SLURM_JOB_ID=328410 SLURM_JOB_NODELIST=c58 SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16 OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
OUT=results/inblock_c58_20261106
set -x
nvcc -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_120,code=\[compute_120,sm_120\] -lineinfo -I src/cuda -c "$OUT/w4a16_naive_original.cu" -o "$OUT/original-naive.o"
c++ -O3 -DNDEBUG build/CMakeFiles/w4a16_bench.dir/src/cuda/bench.cu.o "$OUT/original-naive.o" build/CMakeFiles/w4a16_bench.dir/src/cuda/w4a16_inblock.cu.o build/CMakeFiles/w4a16_bench.dir/src/cuda/dequant.cu.o build/CMakeFiles/w4a16_bench.dir/src/cuda/cublas_gemm.cu.o -o build/w4a16_original_naive_control build/libw4a16_host.a -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lcublas -fopenmp
sha256sum build/w4a16_original_naive_control "$OUT/w4a16_naive_original.cu" > "$OUT/control.sha256"
for variant in original guard; do
  B=./build/w4a16_bench
  if [[ "$variant" == original ]]; then B=./build/w4a16_original_naive_control; fi
  for policy in cold warm; do
    args=()
    if [[ "$policy" == warm ]]; then args+=(--no-flush-l2); fi
    for M in 1 128; do
      prun "$B" --M "$M" --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters 100 --kernels naive "${args[@]}" --csv "$OUT/control_$variant.csv"
    done
  done
done
