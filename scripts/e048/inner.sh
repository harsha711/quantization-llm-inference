#!/usr/bin/env bash
# E047 inner run (one prun process on the node). inner.sh MODE SRC S
set -euo pipefail
MODE=$1 SRC=$2 S=$3
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH
{ date -Is; hostname; nproc; lscpu; nvidia-smi -q; nvcc --version; c++ --version; cmake --version; env | sort | grep -E '^SLURM'; } > "$S/metadata.txt" 2>&1
BUILD_DIR=$SRC/build bash scripts/build.sh > "$S/build.log" 2>&1
cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/flags.make"
ctest --test-dir build --output-on-failure > "$S/ctest.log" 2>&1
build/w4a16_bench --unpack-selftest > "$S/selftest.log" 2>&1
bash scripts/check_inblock.sh > "$S/check.log" 2>&1
bash scripts/e048/check.sh build/w4a16_bench > "$S/check-e048.log" 2>&1
sha256sum build/w4a16_bench src/cuda/*.cu src/cuda/*.cuh > "$S/binaries.sha256"
nvcc -O3 -DNDEBUG -std=c++17 -arch=sm_120 -Xptxas -v -Isrc/cuda -c src/cuda/w4a16_tc_m32.cu -o /dev/null > "$S/ptxas-v.txt" 2>&1
cuobjdump -sass build/w4a16_bench > "$S/sass.txt"
cuobjdump -res-usage build/w4a16_bench > "$S/res-usage.txt" 2>&1
awk '/Function :/{f=$3} /LDG/{c[f]++} /HMMA/{h[f]++} /STG/{s[f]++} /ATOM|RED/{a[f]++} END{for(k in c) print c[k], h[k]+0, s[k]+0, a[k]+0, k}' "$S/sass.txt" | sort -k5 > "$S/ldg_hmma_stg_atom_counts.txt"
K1=tc,tc_m32,cublas_dq
K2=cublas_dq,tc_m32,tc
if [[ $MODE == prereq ]]; then MS="32 128" IT=10 HELD=; else MS="1 4 8 16 32 64 128" IT=100 HELD=1; fi
for ser in 1 2; do ks=K$ser; ks=${!ks}
  for M in $MS; do for pol in flush noflush; do a=(); [[ $pol == noflush ]] && a=(--no-flush-l2)
    build/w4a16_bench --M $M --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters $IT --check-rows 4096 --kernels $ks "${a[@]}" --csv "$S/main-s$ser.csv" >> "$S/main-s$ser.log" 2>&1
  done; done
  if [[ -n $HELD ]]; then for KN in "4096 11008" "11008 4096"; do set -- $KN; for M in 8 32 64 128; do for pol in flush noflush; do a=(); [[ $pol == noflush ]] && a=(--no-flush-l2)
    build/w4a16_bench --M $M --K $1 --N $2 --group 128 --seed 2 --warmup 10 --iters $IT --check-rows 4096 --kernels $ks "${a[@]}" --csv "$S/held-s$ser.csv" >> "$S/held-s$ser.log" 2>&1
  done; done; done; fi
done
echo INNER-OK
