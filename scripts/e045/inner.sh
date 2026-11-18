#!/usr/bin/env bash
# E045 inner run (one prun process on the node). inner.sh MODE SRC S [PREREQ_DIR]
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
# E010's runtime and preserved binaries (historical paths, identities checked below against E010's record).
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 3000 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1

# Runtime: E010's pinned dependency set, restored without resolution only when absent (E005-R2 recipe).
if [[ ! -x $PY ]]; then
  python3.11 -m venv "$VENV"
  PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip \
    "$PY" -m pip install --no-deps -r "$E004/results/e004_20261106/pip-freeze.txt" > "$S/pip-install.log" 2>&1
fi
"$PY" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$E004/results/e004_20261106/pip-freeze.txt" "$S/pip-freeze.txt" > "$S/dependency-diff.txt"
WHEEL=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"]+"/vllm/_C_stable_libtorch.abi3.so")')
# Same library/bridge/wheel/harness bytes as E010 (its verified-inputs record).
sha256sum -c results/e010_c4_20261107/verified-inputs.sha256 > "$S/e010-inputs-check.log"
sha256sum scripts/e008/bench.py scripts/e010/bench.py > "$S/harness.sha256"
grep -q 02f3695023c34db25bc7679f782699e4246d8a4c05b7aaf07db99c28be664759 "$S/harness.sha256"
grep -q bd3c03fff16ad77cfd20c06ea92e5904473565d4d9def1d4b9c226cffb4fed55 "$S/harness.sha256"
base=("$PY" scripts/e045/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")

if [[ $MODE == prereq ]]; then
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run project-gpu bash scripts/check_inblock.sh
  cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/bench-flags.make"
  # tc library: unchanged E044 source and its fallbacks, CMake's Release/-lineinfo/sm_120 device flags.
  run tc-build nvcc -O3 -DNDEBUG -std=c++17 -lineinfo "--generate-code=arch=compute_120,code=[compute_120,sm_120]" \
    -shared -Xcompiler=-fPIC -Isrc/cuda scripts/e045/tc_bridge.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_inblock_vec.cu \
    src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu -o "$S/libe045_tc.so"
  sha256sum "$S/libe045_tc.so" build/w4a16_bench src/cuda/*.cu scripts/e045/* > "$S/binaries.sha256"
  # tc device code must be instruction-identical to the E044 benchmark build.
  sass() { cuobjdump -sass "$1" | awk '/Function :/{p=($3 ~ /w4a16_tc_kernel/)} p'; }
  sass build/w4a16_bench > "$S/sass-tc-bench.txt"; sass "$S/libe045_tc.so" > "$S/sass-tc-lib.txt"
  cuobjdump -res-usage "$S/libe045_tc.so" > "$S/res-usage-lib.txt" 2>&1
  # Same operands as the cycle's seed-3 matrix 0 at M128, all rows checked, E044 benchmark build.
  for KN in "4096 11008" "11008 4096"; do set -- $KN
    run "agree-$1x$2" build/w4a16_bench --M 128 --K "$1" --N "$2" --group 128 --seed 3 --warmup 0 --iters 1 \
      --kernels tc --check-rows 128 --no-flush-l2 --csv "$S/agree.csv"
  done
  base+=(--tc "$S/libe045_tc.so")
  run smoke "${base[@]}" --out "$S/smoke.jsonl" --shapes 256,2048 --warmup 0 --iters 0
  run validation "${base[@]}" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
  run memcheck compute-sanitizer --tool memcheck --error-exitcode 99 "${base[@]}" --out "$S/memcheck.jsonl" --warmup 0 --iters 0
  run synccheck compute-sanitizer --tool synccheck --error-exitcode 99 "${base[@]}" --out "$S/synccheck.jsonl" --warmup 0 --iters 0
  run racecheck compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_tc_kernel \
    "${base[@]}" --out "$S/racecheck.jsonl" --implementations tc --warmup 0 --iters 0
  "$PY" scripts/e045/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/prereq-binary-check.log"
  grep -q "$PRE/libe045_tc.so" "$S/prereq-binary-check.log"
  cp "$PRE/binaries.sha256" "$S/prereq-binaries.sha256"
  base+=(--tc "$PRE/libe045_tc.so")
  clock > "$S/series1-clocks-before.csv"
  run series1 "${base[@]}" --out "$S/series1.jsonl"
  clock > "$S/series1-clocks-after.csv"
  run series2 "${base[@]}" --out "$S/series2.jsonl" --shapes 11008,4096 4096,11008 --ms 128 64 16 4 --series 2
  clock > "$S/series2-clocks-after.csv"
  "$PY" scripts/e045/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
