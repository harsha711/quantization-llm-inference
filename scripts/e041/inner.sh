#!/usr/bin/env bash
# E041 inner run (one prun process on the node). inner.sh MODE SRC S
set -euo pipefail
MODE=$1 SRC=$2 S=$3
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH
{ date -Is; hostname; nproc; taskset -pc $$; lscpu; numactl --hardware; nvidia-smi -q; nvcc --version; c++ --version; cmake --version; env | sort | grep -E '^(SLURM|OMP)'; } > "$S/metadata.txt" 2>&1
[[ $(nproc) == 16 ]] || { echo "expected 16 usable cores, got $(nproc)"; exit 4; }
BUILD_DIR=$SRC/build bash scripts/build.sh > "$S/build.log" 2>&1
ctest --test-dir build --output-on-failure > "$S/ctest.log" 2>&1
cat build/CMakeFiles/w4a16_host.dir/flags.make > "$S/flags.txt"
sha256sum build/cpu_bench build/w4a16_bench build/test_cpu src/cpu_gemm.cpp tools/cpu_bench.cpp > "$S/binaries.sha256"
export OMP_PROC_BIND=close OMP_PLACES=cores
if [[ $MODE == prereq ]]; then PS="1 16" MS=4 REPS=2 WEAK=; GM=4 IT=10; else PS="1 2 4 8 12 16" MS="1 4 128" REPS=10 WEAK=1; GM="1 4 128" IT=100; fi
echo "threads,M,N,K,group,seed,reps,median_ms,min_ms,max_ms,fp64_pass,num_bad,rel_l2,max_abs,fnv1a" > "$S/strong.csv"
cp "$S/strong.csv" "$S/weak.csv"
for M in $MS; do for p in $PS; do OMP_NUM_THREADS=$p build/cpu_bench $M 4096 4096 128 1 $REPS >> "$S/strong.csv" 2>> "$S/strong-samples.log"; done; done
if [[ -n $WEAK ]]; then for p in $PS; do OMP_NUM_THREADS=$p build/cpu_bench 4 $((512*p)) 4096 128 1 $REPS >> "$S/weak.csv" 2>> "$S/weak-samples.log"; done; fi
for M in $GM; do for pol in flush noflush; do a=(); [[ $pol == noflush ]] && a=(--no-flush-l2)
  build/w4a16_bench --M $M --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters $IT --kernels naive,inblock,inblock_m4,cublas "${a[@]}" --csv "$S/gpu.csv" >> "$S/gpu.log" 2>&1
done; done
echo INNER-OK
