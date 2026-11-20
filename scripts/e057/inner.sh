#!/usr/bin/env bash
# E057 inner run (one prun process on the node). inner.sh MODE SRC S [PRE]
#   prereq: build (E041 flags), ctest, check_inblock, -g twin + text identity, STREAM/GPU microbench builds, SASS,
#           perf probe, short versions of every part, compute-sanitizer memcheck of the microbench.
#   full:   binaries of the inspected prerequisite (hash-checked); parts A4, A1, A2, A3, then B.
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH
step() { echo "STEP $* $(date -Is)"; }
step metadata
( set +e; date -Is; date +%z; hostname; nproc; taskset -pc $$; uname -a; lscpu; numactl --hardware; free -g
  head -3 /proc/meminfo; nvidia-smi -q; nvcc --version; c++ --version; gcc --version; cmake --version
  command -v perf; perf --version; echo "perf_event_paranoid $(cat /proc/sys/kernel/perf_event_paranoid)"
  echo "perf_event_max_sample_rate $(cat /proc/sys/kernel/perf_event_max_sample_rate)"
  echo governor; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c
  echo "boost $(cat /sys/devices/system/cpu/cpufreq/boost)"; echo "amd_pstate $(cat /sys/devices/system/cpu/amd_pstate/status)"
  grep MHz /proc/cpuinfo; env | sort | grep -E '^(SLURM|OMP|OMPI)' ) > "$S/metadata.txt" 2>&1
date +%z > "$S/tz.txt"
[[ $(nproc) == 16 ]] || { echo "expected 16 usable cores, got $(nproc)"; exit 4; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]

if [[ $MODE == prereq ]]; then
  step build
  BUILD_DIR=$SRC/build CUDA_ARCH=120 bash scripts/build.sh > "$S/build.log" 2>&1
  ctest --test-dir build --output-on-failure > "$S/ctest.log" 2>&1
  bash scripts/check_inblock.sh > "$S/check_inblock.log" 2>&1
  cat build/CMakeFiles/w4a16_host.dir/flags.make > "$S/flags.txt"
  # -g twin of cpu_bench for source-line attribution only (same -O3 flags plus -g)
  cmake -S . -B build-g -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g -DCMAKE_CUDA_ARCHITECTURES=120 > "$S/build-g.log" 2>&1
  cmake --build build-g --target cpu_bench -j16 >> "$S/build-g.log" 2>&1
  cat build-g/CMakeFiles/w4a16_host.dir/flags.make > "$S/flags-g.txt"
  objdump -d --no-show-raw-insn build/cpu_bench | tail -n +4 > "$S/text-O3.txt"
  objdump -d --no-show-raw-insn build-g/cpu_bench | tail -n +4 > "$S/text-g.txt"
  if cmp -s "$S/text-O3.txt" "$S/text-g.txt"; then echo IDENTICAL; else echo DIFFERENT; fi > "$S/text-identity.txt"
  gcc -O3 -std=gnu99 -g -Wno-unused-result -fopenmp -o build/stream scripts/e057/stream.c -lm 2> "$S/stream-build.log"
  nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 -o build/e057_gpu scripts/e057/e057_gpu.cu > "$S/gpu-build.log" 2>&1
  mkdir "$S/bin"; cp build/cpu_bench build/stream build/e057_gpu "$S/bin/"; cp build-g/cpu_bench "$S/bin/cpu_bench_g"
  (cd "$S" && sha256sum bin/* > bin.sha256)
  BIN=$S/bin
fi
sha256sum src/cpu_gemm.cpp tools/cpu_bench.cpp scripts/e057/stream.c scripts/e057/e057_gpu.cu > "$S/sources.sha256"
if [[ $MODE == full ]]; then
  step prereq-binaries
  (cd "$PRE" && sha256sum -c bin.sha256) > "$S/prereq-binary-check.log"
  diff "$PRE/sources.sha256" "$S/sources.sha256"   # binaries were built from these exact sources
  BIN=$PRE/bin
fi
cuobjdump -sass "$BIN/e057_gpu" > "$S/sass.txt"
grep -o 'HMMA[.A-Z0-9]*' "$S/sass.txt" | sort | uniq -c > "$S/sass-hmma.txt"
[[ -s "$S/sass-hmma.txt" ]]
cuobjdump -res-usage "$BIN/e057_gpu" > "$S/res-usage.txt" 2>&1 || true

export OMP_PROC_BIND=close OMP_PLACES=cores
# ---------------- perf availability (recorded; parts A1/A2 are skipped if unavailable) ----------------
step perf-probe
mkdir -p "$S/perf"
PERF_OK=0
if command -v perf > /dev/null 2>&1 && perf stat -x, -e cycles:u,instructions:u -o "$S/perf/probe.csv" true > "$S/perf/probe.out" 2>&1 \
   && grep -q ',cycles:u,' "$S/perf/probe.csv" && ! grep -q 'not supported\|not counted' "$S/perf/probe.csv"; then
  PERF_OK=1
  perf list > "$S/perf/perf-list.txt" 2>&1 || true
  perf stat -vv -e cycles:u,instructions:u,L1-dcache-loads:u,L1-dcache-load-misses:u,cache-references:u,cache-misses:u,LLC-loads:u,LLC-load-misses:u,stalled-cycles-frontend:u,stalled-cycles-backend:u true \
    > /dev/null 2> "$S/perf/event-mapping.txt" || true
else
  { echo "UNAVAILABLE"; command -v perf || echo "perf: not on PATH"; cat "$S/perf/probe.out" "$S/perf/probe.csv" 2>/dev/null; } > "$S/perf/UNAVAILABLE.txt"
fi
echo "$PERF_OK" > "$S/perf/perf_ok.txt"

if [[ $MODE == prereq ]]; then MS=4; PS="1 16"; SREPS=1; SPS="1 16"; RREPS=2
else MS="1 4 128"; PS="1 2 4 8 12 16"; SREPS=5; SPS="1 2 4 8 12 16"; RREPS=10; fi
hi() { # calls in the "hi" run of the differencing pair: lo = 2
  local M=$1 p=$2
  if [[ $MODE == prereq ]]; then echo 4; elif [[ $M == 128 ]]; then echo $((2 + 5 * p)); else echo $((2 + 20 * p)); fi; }
cb() { "$BIN/cpu_bench" "$1" 4096 4096 128 1 "$2"; }

# ---------------- A4: E041 re-time (no perf) ----------------
step A4
echo "threads,M,N,K,group,seed,reps,median_ms,min_ms,max_ms,fp64_pass,num_bad,rel_l2,max_abs,fnv1a" > "$S/a4-retime.csv"
for M in $MS; do for p in 1 16; do OMP_NUM_THREADS=$p cb $M $RREPS >> "$S/a4-retime.csv" 2>> "$S/a4-samples.log"; done; done

# ---------------- A1: perf stat at T1 (differencing: (hi - lo) / (hi - lo calls)) ----------------
G1=cycles:u,instructions:u,task-clock
G2=L1-dcache-loads:u,L1-dcache-load-misses:u
G3=cache-references:u,cache-misses:u
G4=LLC-loads:u,LLC-load-misses:u
G5=stalled-cycles-frontend:u,stalled-cycles-backend:u
pstat() { # tag events M p R
  OMP_NUM_THREADS=$4 perf stat -x, -r 3 -e "$2" -o "$S/perf/$1.csv" "$BIN/cpu_bench" "$3" 4096 4096 128 1 "$5" >> "$S/perf/$1.out" 2>&1; }
if [[ $PERF_OK == 1 ]]; then
  step A1
  for M in $MS; do H=$(hi $M 1); for R in 2 $H; do
    i=0; for G in "$G1" "$G2" "$G3" "$G4" "$G5"; do i=$((i+1)); pstat "a1-M$M-p1-R$R-g$i" "$G" $M 1 $R; done
  done; done
  step A1-record
  for M in $MS; do H=$(hi $M 1)
    OMP_NUM_THREADS=1 perf record -e cycles:u -F 1999 -o "$S/perf/rec-M$M.data" "$BIN/cpu_bench_g" $M 4096 4096 128 1 $H > "$S/perf/rec-M$M.out" 2>&1
    perf report -i "$S/perf/rec-M$M.data" --stdio --no-children --sort dso,symbol > "$S/perf/rec-M$M-symbol.txt" 2>&1
    perf report -i "$S/perf/rec-M$M.data" --stdio --no-children --sort srcline > "$S/perf/rec-M$M-srcline.txt" 2>&1
    perf report -i "$S/perf/rec-M$M.data" --stdio --no-children --sort symbol,srcline -t '|' > "$S/perf/rec-M$M-symsrc.txt" 2>&1
    perf annotate -i "$S/perf/rec-M$M.data" --stdio -l > "$S/perf/rec-M$M-annotate.txt" 2>&1 || true
  done
  step A2
  for M in $MS; do for p in $PS; do H=$(hi $M $p); for R in 2 $H; do pstat "a2-M$M-p$p-R$R" "$G1" $M $p $R; done; done; done
fi

# ---------------- A3: STREAM (midterm source, unchanged; 64M doubles per array) ----------------
step A3
mkdir -p "$S/stream"
for p in $SPS; do for r in $(seq 1 $SREPS); do OMP_NUM_THREADS=$p "$BIN/stream" > "$S/stream/t$p-r$r.txt" 2>&1; done; done

# ---------------- B: GPU microbenchmarks ----------------
step B
clockq() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,temperature.gpu,power.draw,power.limit --format=csv; }
clockq > "$S/b-clk-before.csv"; nvidia-smi -q -d PERFORMANCE,CLOCK > "$S/b-perf-before.txt" 2>&1
if [[ $MODE == prereq ]]; then
  compute-sanitizer --version > "$S/sanitizer-version.txt" 2>&1
  compute-sanitizer --tool memcheck --error-exitcode 9 "$BIN/e057_gpu" --tiny > "$S/b-memcheck.log" 2>&1
  GARG=--small
else GARG=; fi
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -lms 100 > "$S/b-clk-during.csv" 2>&1 &
SMI=$!
rc=0; "$BIN/e057_gpu" $GARG > "$S/b-gpu.txt" 2> "$S/b-gpu-samples.log" || rc=$?
kill $SMI || true; wait $SMI || true
clockq > "$S/b-clk-after.csv"; nvidia-smi -q -d PERFORMANCE,CLOCK > "$S/b-perf-after.txt" 2>&1
[[ $rc == 0 ]]
step analyze
python3 "$SRC/scripts/e057/analyze.py" "$S" --mode "$MODE" > "$S/analysis.json" 2> "$S/analysis.txt"
echo INNER-OK
