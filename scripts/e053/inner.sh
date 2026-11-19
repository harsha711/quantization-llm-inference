#!/usr/bin/env bash
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
[[ $SRC == /* && -f $SRC/CMakeLists.txt && $PROJECT_ROOT == "$SRC" ]]
cd "$SRC"
unset W4A16_V1_SPLITS W4A16_V2_SPLITS
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
run() { local label=$1 rc=0; shift; (set -x; timeout 2100 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version; mpirun --version; compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1
[[ $(nproc) -ge 16 ]]
[[ -x $PY ]]  # Stop rather than substitute dependencies or install a new environment silently.
"$PY" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$E004/results/e004_20261106/pip-freeze.txt" "$S/pip-freeze.txt" > "$S/dependency-diff.txt"
sha256sum -c results/e010_c4_20261107/verified-inputs.sha256 > "$S/e010-inputs-check.log"
WHEEL=$(find /tmp/e005-r2-hpuvvad-20261106/lib -path '*/vllm/_C_stable_libtorch.abi3.so' -print -quit)
[[ -f $WHEEL ]]
base=("$PY" scripts/e053/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")
APP=(--replay-mode application --app-replay-mode strict --app-replay-match grid --profile-from-start off --graph-profiling node --cache-control none --clock-control none --pipeline-boost-state dynamic --launch-skip 399 --launch-count 1)
profile() {
  local name=$1 path=$2
  mkdir -p "$S/ncu"
  run "$name" ncu "${APP[@]}" --metrics "$(paste -sd, scripts/e053/metrics.txt)" --export "$S/ncu/$name" \
    "${base[@]}" --tc "$LIB" --profile --implementations "$path" --ms 4 --shapes 4096,4096 --warmup 0 --iters 0 --out "$S/ncu/$name.jsonl"
  ncu --import "$S/ncu/$name.ncu-repz" --page raw --csv --print-units base > "$S/ncu/$name.raw.csv"
}
if [[ $MODE != full ]]; then
  if [[ $MODE == baseline && -n $PRE ]]; then
    [[ -e $PRE/BUILD_REUSE_INSPECTED ]]
    (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/build-reuse-check.log"
    grep -q "$PRE/libe053.so: OK" "$S/build-reuse-check.log"
    printf 'Reused exact checked binaries and gates from %s; prior driver failure was Python import shadowing.\n' "$PRE" > "$S/build-reuse.txt"
    for tag in project-build project-ctest project-gpu v3-check library-build; do
      [[ $(cat "$PRE/$tag.exit") == 0 ]]
      cp "$PRE/$tag.log" "$PRE/$tag.exit" "$S/"
    done
    for file in libe053.so w4a16_bench bench-flags.make sass-library.txt sass-bench.txt resources.txt; do cp "$PRE/$file" "$S/"; done
    LIB=$S/libe053.so
    sha256sum "$LIB" "$S/w4a16_bench" src/cuda/*.cu src/cuda/*.cuh scripts/e053/* > "$S/binaries.sha256"
  else
    run project-build bash scripts/build.sh
    run project-ctest ctest --test-dir build --output-on-failure
    run project-gpu bash scripts/check_inblock.sh
    run v3-check env KERNELS=v3 bash scripts/ours/check.sh
    files=(src/cuda/ours_v1.cu src/cuda/ours_v2.cu src/cuda/ours_v3.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_tc_m32.cu src/cuda/w4a16_inblock_vec.cu src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu)
    defs=()
    [[ $MODE == baseline ]] || { defs=(-DE053_CANDIDATE); files+=(src/cuda/ours_v3_m2.cu); }
    LIB=$S/libe053.so
    run library-build nvcc -O3 -DNDEBUG -std=c++17 -lineinfo '--generate-code=arch=compute_120,code=[compute_120,sm_120]' \
      -shared -Xcompiler=-fPIC -Isrc/cuda "${defs[@]}" scripts/e053/bridge.cu "${files[@]}" -o "$LIB"
    cp build/w4a16_bench "$S/w4a16_bench"
    cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/bench-flags.make"
    sha256sum "$LIB" "$S/w4a16_bench" src/cuda/*.cu src/cuda/*.cuh scripts/e053/* > "$S/binaries.sha256"
    cuobjdump -sass "$LIB" > "$S/sass-library.txt"
    cuobjdump -sass "$S/w4a16_bench" > "$S/sass-bench.txt"
    cuobjdump -res-usage "$LIB" > "$S/resources.txt"
  fi
  paths=(v3 tc_ladder marlin cublas_dq)
  [[ $MODE == baseline ]] || paths=(v3 v3_m2 tc_ladder marlin cublas_dq)
  run smoke "${base[@]}" --tc "$LIB" --out "$S/smoke.jsonl" --implementations "${paths[@]}" --shapes 256,256 --ms 4 --trace --warmup 0 --iters 0
  if [[ $MODE == baseline ]]; then
    ncu --query-metrics > "$S/ncu-query-metrics.txt" 2>&1
    while read -r metric; do [[ $metric == launch__* ]] && continue; grep -q "^${metric%%.*}[[:space:]]" "$S/ncu-query-metrics.txt"; done < scripts/e053/metrics.txt
    run calibration ncu "${APP[@]}" --metrics gpu__time_duration.sum,launch__grid_size --export "$S/calibration" "$PY" scripts/e049/calibrate.py
    ncu --import "$S/calibration.ncu-repz" --page raw --csv --print-units base > "$S/calibration.raw.csv"
    profile baseline-v3 v3
  else
    run candidate-check bash scripts/e053/check.sh
    run validation "${base[@]}" --tc "$LIB" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
    for tool in memcheck racecheck synccheck; do
      run "cycle-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 --kernel-name kns=w4a16_v3 \
        "${base[@]}" --tc "$LIB" --out "$S/cycle-$tool.jsonl" --implementations v3 v3_m2 --shapes 256,256 --ms 1 2 5 16 --warmup 0 --iters 0
    done
  fi
else
  [[ -e $PRE/INSPECTED && $(cat "$PRE/driver.exit") == 0 ]]
  (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/prereq-binary-check.log"
  grep -q "$PRE/libe053.so: OK" "$S/prereq-binary-check.log"
  LIB=$PRE/libe053.so
  clock > "$S/clocks-before.csv"
  run series1 "${base[@]}" --tc "$LIB" --out "$S/series1.jsonl"
  run series2 "${base[@]}" --tc "$LIB" --out "$S/series2.jsonl" --shapes 11008,4096 4096,11008 4096,4096 --ms 16 8 4 2 1 --series 2
  clock > "$S/clocks-after.csv"
  for series in 1 2; do
    kernels=v3,v3_m2,tc_ladder; ms='1 2 4 8 16'
    [[ $series == 1 ]] || { kernels=tc_ladder,v3_m2,v3; ms='16 8 4 2 1'; }
    for kn in '4096 4096' '1536 1536' '1536 8960' '8960 1536'; do
      read -r k n <<< "$kn"
      for m in $ms; do for policy in read reused; do
        flag=--flush-read; [[ $policy == read ]] || flag=--no-flush-l2
        run "single-s$series-K$k-N$n-M$m-$policy" "$PRE/w4a16_bench" --M "$m" --N "$n" --K "$k" --group 128 --seed 1 \
          --warmup 10 --iters 200 --check-rows 4096 --kernels "$kernels" "$flag" --csv "$S/single-s$series.csv"
      done; done
    done
  done
  profile final-v3 v3
  profile final-v3-m2 v3_m2
fi
echo INNER-OK
