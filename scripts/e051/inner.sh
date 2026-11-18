#!/usr/bin/env bash
# E051 inner run (one prun process on the node). inner.sh MODE SRC S [PREREQ_DIR]
# Descriptive replays of `tc_ladder` (E050): (a) single-matrix bench sweep, (b) E009 cycle, (c) E041 same-node row,
# (d) E049-method cyclic profiles. Runtime/identity block copied from scripts/e050/inner.sh.
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 5400 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; command -v ncu; ncu --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1
[[ $(nproc) == 16 ]]

if [[ ! -x $PY ]]; then
  echo "venv absent: restoring" > "$S/venv-restored.txt"
  python3.11 -m venv "$VENV"
  PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip \
    "$PY" -m pip install --no-deps -r "$E004/results/e004_20261106/pip-freeze.txt" > "$S/pip-install.log" 2>&1
fi
"$PY" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$E004/results/e004_20261106/pip-freeze.txt" "$S/pip-freeze.txt" > "$S/dependency-diff.txt"
WHEEL=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
sha256sum -c results/e010_c4_20261107/verified-inputs.sha256 > "$S/e010-inputs-check.log"
sha256sum scripts/e008/bench.py scripts/e010/bench.py > "$S/harness.sha256"
grep -q 02f3695023c34db25bc7679f782699e4246d8a4c05b7aaf07db99c28be664759 "$S/harness.sha256"
grep -q bd3c03fff16ad77cfd20c06ea92e5904473565d4d9def1d4b9c226cffb4fed55 "$S/harness.sha256"
ext=(--source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")
cyc=("$PY" scripts/e051/cyclic.py "${ext[@]}")
prof=("$PY" scripts/e049/bench.py "${ext[@]}")
B=build/w4a16_bench
APP=(--replay-mode application --app-replay-mode strict --app-replay-match grid --profile-from-start off
     --graph-profiling node --cache-control none --clock-control none --pipeline-boost-state dynamic)
mkdir -p "$S/ncu"
# profile TAG IMPL M K N: E049's unchanged profile mode (fourth GEMM of the last of 100 graph cycles).
# tc_ladder at M >= 64 is the unchanged tc_m32 launcher (scripts/e050/tc_bridge.cu), so IMPL tc_m32 is that path.
profile() {
  local tag=$1 impl=$2 m=$3 k=$4 n=$5
  clock > "$S/ncu/$tag-clocks-before.csv"
  run "profile_$tag" ncu "${APP[@]}" --launch-skip 399 --launch-count 1 \
    --section LaunchStats --section Occupancy --section InstructionStats --metrics "$(cat "$S/metrics-used.txt")" \
    --export "$S/ncu/$tag" "${prof[@]}" --tc "$LIB49" --profile --implementations "$impl" --ms "$m" --shapes "$k,$n" \
    --warmup 0 --iters 0 --out "$S/ncu/$tag.jsonl"
  clock > "$S/ncu/$tag-clocks-after.csv"
  ncu --import "$S/ncu/$tag.ncu-repz" --page raw --csv --print-units base > "$S/ncu/$tag.raw.csv"
  ncu --import "$S/ncu/$tag.ncu-repz" --page raw --csv --print-units base --print-metric-instances values \
    > "$S/ncu/$tag.instances.csv" 2> "$S/ncu/$tag.instances.err" || echo "instances export rc=$?" >> "$S/ncu/$tag.instances.err"
  ncu --import "$S/ncu/$tag.ncu-repz" --page details --csv --print-units base > "$S/ncu/$tag.details.csv"
}
pol() { [[ $1 == noflush ]] && echo --no-flush-l2 || true; }
K1=naive,inblock,inblock_specialized,inblock_m2,inblock_m4,inblock_vec,inblock_m4_vec,tc,tc_m32,tc_ladder,cublas,cublas_dq,dequant,unfused
K2=$(tr , '\n' <<< "$K1" | tac | paste -sd,)
H1=inblock_m4_vec,tc,tc_m32,tc_ladder,cublas_dq
H2=$(tr , '\n' <<< "$H1" | tac | paste -sd,)

if [[ $MODE == prereq ]]; then
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run unpack-selftest $B --unpack-selftest
  run project-gpu bash scripts/check_inblock.sh
  run e048-check bash scripts/e048/check.sh $B
  # tc_ladder selector: both sides of the threshold, tails, small group; bench checks all rows and bitwise == tc.
  for M in 1 5 63 64 65 128; do
    run "ladder-check-M$M" $B --M $M --N 96 --K 256 --group 64 --seed 3 --warmup 1 --iters 2 --check-rows 4096 --kernels tc,tc_ladder
  done
  run ladder-memcheck compute-sanitizer --tool memcheck --error-exitcode 99 \
    $B --M 128 --N 4096 --K 4096 --group 128 --seed 1 --warmup 0 --iters 1 --kernels tc_ladder --no-flush-l2
  cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/bench-flags.make"
  # Libraries: E050's bridge (cycle) and E049's bridge (profile mode), unchanged sources, E049/E050's exact nvcc command.
  for v in e050 e049; do
    run "lib-$v" nvcc -O3 -DNDEBUG -std=c++17 -lineinfo "--generate-code=arch=compute_120,code=[compute_120,sm_120]" \
      -shared -Xcompiler=-fPIC -Isrc/cuda scripts/$v/tc_bridge.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_tc_m32.cu \
      src/cuda/w4a16_inblock_vec.cu src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu -o "$S/lib${v}_tc.so"
  done
  cp build/w4a16_bench build/cpu_bench "$S/"
  sha256sum "$S"/lib*_tc.so "$S/w4a16_bench" "$S/cpu_bench" src/cuda/*.cu src/cuda/*.cuh scripts/e051/* scripts/e049/* scripts/e050/tc_bridge.cu > "$S/binaries.sha256"
  sass() { cuobjdump -sass "$1" | awk '/Function :/{p=($3 ~ /w4a16_tc_(m32_)?kernel/)} p'; }
  sass $B > "$S/sass-tc-bench.txt"; sass "$S/libe050_tc.so" > "$S/sass-tc-lib50.txt"; sass "$S/libe049_tc.so" > "$S/sass-tc-lib49.txt"
  # (a) smoke: whole kernel list, both policies, one M per side of the threshold.
  for M in 5 128; do for p in flush noflush; do
    run "a-smoke-M$M-$p" $B --M $M --N 4096 --K 4096 --group 128 --seed 1 --warmup 1 --iters 2 --check-rows 4096 --kernels $K1 $(pol $p) --csv "$S/a-smoke.csv"
  done; done
  # (c) smoke: CPU T1/T16 one rep (E041 binding).
  for p in 1 16; do OMP_NUM_THREADS=$p OMP_PROC_BIND=close OMP_PLACES=cores build/cpu_bench 4 4096 4096 128 1 1 >> "$S/c-smoke.csv"; done
  # (b) cycle: smoke, full validation, sanitizers (E009 recipe: unfiltered memcheck/synccheck, racecheck on our kernels).
  run b-smoke "${cyc[@]}" --tc "$S/libe050_tc.so" --out "$S/b-smoke.jsonl" --shapes 256,2048 --warmup 0 --iters 0
  run b-validation "${cyc[@]}" --tc "$S/libe050_tc.so" --out "$S/b-validation.jsonl" --warmup 0 --iters 0
  run b-memcheck compute-sanitizer --tool memcheck --error-exitcode 99 "${cyc[@]}" --tc "$S/libe050_tc.so" --out "$S/b-memcheck.jsonl" --warmup 0 --iters 0
  run b-synccheck compute-sanitizer --tool synccheck --error-exitcode 99 "${cyc[@]}" --tc "$S/libe050_tc.so" --out "$S/b-synccheck.jsonl" --warmup 0 --iters 0
  run b-racecheck compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_tc \
    "${cyc[@]}" --tc "$S/libe050_tc.so" --out "$S/b-racecheck.jsonl" --implementations tc_ladder --warmup 0 --iters 0
  # (d) metric availability (E046 rule), E049 window calibration, one real profile.
  ncu --query-metrics > "$S/ncu-query-metrics.txt" 2>&1
  : > "$S/metrics-dropped.txt"; keep=()
  while read -r m; do [[ -z $m ]] && continue
    if grep -q "^${m%%.*}[[:space:]]" "$S/ncu-query-metrics.txt"; then keep+=("$m"); else echo "$m" >> "$S/metrics-dropped.txt"; fi
  done < scripts/e049/metrics.txt
  [[ ! -s $S/metrics-dropped.txt ]]
  (IFS=,; echo "${keep[*]}") > "$S/metrics-used.txt"
  run calibration ncu "${APP[@]}" --launch-skip 399 --launch-count 1 \
    --metrics gpu__time_duration.sum,launch__grid_size --export "$S/ncu/calibration" "$PY" scripts/e049/calibrate.py
  ncu --import "$S/ncu/calibration.ncu-repz" --page raw --csv --print-units base > "$S/ncu/calibration.raw.csv"
  LIB49=$S/libe049_tc.so
  profile tc_ladder_M64_K4096_N11008 tc_m32 64 4096 11008
  "$PY" scripts/e051/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/prereq-binary-check.log"
  grep -q "$PRE/libe050_tc.so: OK" "$S/prereq-binary-check.log"; grep -q "$PRE/libe049_tc.so: OK" "$S/prereq-binary-check.log"
  cp "$PRE/binaries.sha256" "$S/prereq-binaries.sha256"; cp "$PRE/metrics-used.txt" "$S/metrics-used.txt"
  # The prerequisite's own binaries (hash-checked above), so every timed point uses the inspected executable.
  grep -q "$PRE/w4a16_bench: OK" "$S/prereq-binary-check.log"; grep -q "$PRE/cpu_bench: OK" "$S/prereq-binary-check.log"
  B=$PRE/w4a16_bench
  LIB49=$PRE/libe049_tc.so
  clock > "$S/a-clocks-before.csv"
  # (a) single-matrix sweep, E001-E003/E043/E044 conditions; series 2 reverses kernel and M order.
  for ser in 1 2; do ks=K$ser; ks=${!ks}; hs=H$ser; hs=${!hs}
    MS="1 5 8 32 64 128"; [[ $ser == 2 ]] && MS="128 64 32 8 5 1"
    for M in $MS; do for p in flush noflush; do
      run "a-main-s$ser-M$M-$p" $B --M $M --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters 100 --check-rows 4096 --kernels $ks $(pol $p) --csv "$S/a-main-s$ser.csv"
    done; done
    HM="8 32 64 128"; [[ $ser == 2 ]] && HM="128 64 32 8"
    for KN in "4096 11008" "11008 4096"; do set -- $KN; for M in $HM; do for p in flush noflush; do
      run "a-held-s$ser-K$1-M$M-$p" $B --M $M --K $1 --N $2 --group 128 --seed 2 --warmup 10 --iters 100 --check-rows 4096 --kernels $hs $(pol $p) --csv "$S/a-held-s$ser.csv"
    done; done; done
  done
  clock > "$S/a-clocks-after.csv"
  # (c) E041 row on this node: CPU T1/T16 (E041 binding), then E041's GPU list plus tc_ladder.
  echo "threads,M,N,K,group,seed,reps,median_ms,min_ms,max_ms,fp64_pass,num_bad,rel_l2,max_abs,fnv1a" > "$S/c-cpu.csv"
  for M in 1 4 128; do for p in 1 16; do
    OMP_NUM_THREADS=$p OMP_PROC_BIND=close OMP_PLACES=cores "$PRE/cpu_bench" $M 4096 4096 128 1 10 >> "$S/c-cpu.csv" 2>> "$S/c-cpu-samples.log"
  done; done
  for M in 1 4 128; do for p in flush noflush; do
    run "c-gpu-M$M-$p" $B --M $M --N 4096 --K 4096 --group 128 --seed 1 --warmup 10 --iters 100 --kernels naive,inblock,inblock_m4,cublas,tc_ladder $(pol $p) --csv "$S/c-gpu.csv"
  done; done
  # (b) E009 cycle with tc_ladder; series 2 reverses shapes, counts and path order.
  clock > "$S/b-clocks-before.csv"
  run b-series1 "${cyc[@]}" --tc "$PRE/libe050_tc.so" --out "$S/b-series1.jsonl"
  run b-series2 "${cyc[@]}" --tc "$PRE/libe050_tc.so" --out "$S/b-series2.jsonl" --shapes 11008,4096 4096,11008 --counts 4 2 1 --series 2
  clock > "$S/b-clocks-after.csv"
  # (d) tc_ladder (= tc_m32 launcher) at M64/M128 and Marlin at M128 (E049 profiled Marlin at M4/M64 only).
  for KN in "4096 11008" "11008 4096"; do set -- $KN
    profile "tc_ladder_M64_K$1_N$2" tc_m32 64 "$1" "$2"
    profile "tc_ladder_M128_K$1_N$2" tc_m32 128 "$1" "$2"
    profile "marlin_M128_K$1_N$2" marlin 128 "$1" "$2"
  done
  "$PY" scripts/e051/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
