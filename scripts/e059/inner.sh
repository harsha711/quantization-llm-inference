#!/usr/bin/env bash
# E059 inner run (one prun process on the node). inner.sh MODE SRC S [PREREQ_DIR]
# Copy of E050's inner.sh: E010 runtime/identity checks, project checks, one library (E050's unchanged tc/tc_m32
# build + E047's unchanged tc_splitk through a forced-S launcher + the teammate's unchanged v1/v2/v3 and read flush).
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
unset W4A16_V1_SPLITS W4A16_V2_SPLITS
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 9000 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI|W4A16)'; } > "$S/metadata.txt" 2>&1

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
sha256sum -c scripts/e059/teammate_src.sha256 > "$S/teammate-src-check.log"
base=("$PY" scripts/e059/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")

if [[ $MODE == prereq ]]; then
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run project-gpu bash scripts/check_inblock.sh
  run e047-check bash scripts/e047/check.sh build/w4a16_bench
  cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/bench-flags.make"
  # One library: E050's exact source list and nvcc command, plus the E059 launcher/bridge and the teammate's files.
  run lib-build nvcc -O3 -DNDEBUG -std=c++17 -lineinfo "--generate-code=arch=compute_120,code=[compute_120,sm_120]" \
    -shared -Xcompiler=-fPIC -Isrc/cuda scripts/e050/tc_bridge.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_tc_m32.cu \
    src/cuda/w4a16_inblock_vec.cu src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu \
    scripts/e059/splitk_forced.cu scripts/e059/ours_bridge.cu scripts/e059/teammate_src/ours_v1.cu \
    scripts/e059/teammate_src/ours_v2.cu scripts/e059/teammate_src/ours_v3.cu scripts/e059/teammate_src/flush_read.cu \
    -Xptxas -v -o "$S/libe059.so"
  sha256sum "$S/libe059.so" build/w4a16_bench src/cuda/*.cu src/cuda/*.cuh scripts/e050/tc_bridge.cu scripts/e059/*.* \
    scripts/e059/teammate_src/* > "$S/binaries.sha256"
  sass() { cuobjdump -sass "$1" | awk -v re="$2" '/Function :/{p=($3 ~ re)} p'; }
  sass build/w4a16_bench 'w4a16_tc_(m32_)?kernel' > "$S/sass-tc-bench.txt"; sass "$S/libe059.so" 'w4a16_tc_(m32_)?kernel' > "$S/sass-tc-lib.txt"
  sass build/w4a16_bench 'w4a16_tc_splitk_kernel' > "$S/sass-splitk-bench.txt"; sass "$S/libe059.so" 'w4a16_tc_splitk_kernel' > "$S/sass-splitk-lib.txt"
  sass "$S/libe059.so" 'w4a16_v[123]_kernel' > "$S/sass-ours-lib.txt"
  cp results/e050_20261118/prereq-c8-343302/sass-tc-lib.txt "$S/sass-tc-e050.txt"
  cuobjdump -res-usage "$S/libe059.so" > "$S/res-usage-lib.txt" 2>&1
  base+=(--lib "$S/libe059.so")
  # Contract (FP64, every row) of every path incl. the teammate's v1/v2 on every shape, 2 matrices, traced.
  run validation "${base[@]}" --out "$S/validation.jsonl" --count 2 --extra v1 v2 --trace --iters 0 --rf-iters 0
  # Smoke timing at M = 1 with the planned matrix counts (inspection only, not a result).
  run smoke "${base[@]}" --out "$S/smoke.jsonl" --shapes 11008,4096 8960,1536 1536,1536 --ms 1 --warmup 2 --iters 20 \
    --rf-warmup 2 --rf-iters 20
  # One cycle (+ one read-flush sample) of every path under memcheck/synccheck, 2 matrices, a part-(b) and a model shape.
  for tool in memcheck synccheck; do
    run "$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 "${base[@]}" --out "$S/$tool.jsonl" \
      --shapes 8960,1536 1536,1536 --count 2 --warmup 0 --iters 1 --rf-warmup 0 --rf-iters 1
  done
  run racecheck-v3 compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_v3 "${base[@]}" \
    --out "$S/racecheck-v3.jsonl" --shapes 8960,1536 --ms 1 16 --count 1 --implementations v3 --iters 0 --rf-iters 0
  run racecheck-splitk compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_tc_splitk "${base[@]}" \
    --out "$S/racecheck-splitk.jsonl" --shapes 8960,1536 --ms 1 16 --count 1 \
    --implementations splitk_s1 splitk_s2 splitk_s3 splitk_s4 splitk_s8 --iters 0 --rf-iters 0
  "$PY" scripts/e059/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/prereq-binary-check.log"
  grep -q "$PRE/libe059.so: OK" "$S/prereq-binary-check.log"
  cp "$PRE/binaries.sha256" "$S/prereq-binaries.sha256"
  # Device sources of this job equal the prerequisite's (the library itself is reused, not rebuilt).
  grep -E '  (src/cuda/|scripts/e050/tc_bridge.cu|scripts/e059/[a-z_]+\.cu|scripts/e059/teammate_src/)' "$PRE/binaries.sha256" \
    > "$S/device-sources.sha256"
  [[ $(wc -l < "$S/device-sources.sha256") -ge 20 ]]
  sha256sum -c "$S/device-sources.sha256" > "$S/device-sources-check.log"
  base+=(--lib "$PRE/libe059.so")
  clock > "$S/series1-clocks-before.csv"
  run series1 "${base[@]}" --out "$S/series1.jsonl"
  clock > "$S/series1-clocks-after.csv"
  run series2 "${base[@]}" --out "$S/series2.jsonl" --shapes 11008,4096 4096,11008 1536,1536 8960,1536 1536,8960 \
    --ms 16 4 1 --series 2
  clock > "$S/series2-clocks-after.csv"
  "$PY" scripts/e059/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
