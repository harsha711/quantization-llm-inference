#!/usr/bin/env bash
# E050 inner run (one prun process on the node). inner.sh MODE SRC S [PREREQ_DIR]
# Copy of E049's inner.sh without profiles: tc_ladder replaces tc_m32, three shapes, seeds 7-10.
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
run() { local label=$1 rc=0; shift; (set -x; timeout 5400 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1

# Runtime: E010's pinned dependency set, restored without resolution only when absent (E005-R2 recipe, as E045).
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
base=("$PY" scripts/e050/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")

if [[ $MODE == prereq ]]; then
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run project-gpu bash scripts/check_inblock.sh
  run e048-check bash scripts/e048/check.sh build/w4a16_bench
  cp build/CMakeFiles/w4a16_bench.dir/flags.make "$S/bench-flags.make"
  # Library: unchanged E044/E048 sources and fallbacks + the E050 bridge, E049's exact nvcc command.
  run tc-build nvcc -O3 -DNDEBUG -std=c++17 -lineinfo "--generate-code=arch=compute_120,code=[compute_120,sm_120]" \
    -shared -Xcompiler=-fPIC -Isrc/cuda scripts/e050/tc_bridge.cu src/cuda/w4a16_tc.cu src/cuda/w4a16_tc_m32.cu \
    src/cuda/w4a16_inblock_vec.cu src/cuda/w4a16_inblock_rows.cu src/cuda/w4a16_inblock_specialized.cu -o "$S/libe050_tc.so"
  sha256sum "$S/libe050_tc.so" build/w4a16_bench src/cuda/*.cu src/cuda/*.cuh scripts/e050/* > "$S/binaries.sha256"
  # Device code of both kernels must match the same-job bench build and E049's library (analyzer).
  sass() { cuobjdump -sass "$1" | awk '/Function :/{p=($3 ~ /w4a16_tc_(m32_)?kernel/)} p'; }
  sass build/w4a16_bench > "$S/sass-tc-bench.txt"; sass "$S/libe050_tc.so" > "$S/sass-tc-lib.txt"
  cp results/e049_20261117/prereq-c8-343274/sass-tc-lib.txt "$S/sass-tc-e049.txt"
  cuobjdump -res-usage "$S/libe050_tc.so" > "$S/res-usage-lib.txt" 2>&1
  # Same operands as the cycle's seed-7 matrix 0 at M128, all rows checked (bench also checks tc_m32 == tc bitwise).
  for KN in "4096 11008" "11008 4096" "4096 4096"; do set -- $KN
    run "agree-$1x$2" build/w4a16_bench --M 128 --K "$1" --N "$2" --group 128 --seed 7 --warmup 0 --iters 1 \
      --kernels tc,tc_m32 --check-rows 128 --no-flush-l2 --csv "$S/agree.csv"
  done
  base+=(--tc "$S/libe050_tc.so")
  run smoke "${base[@]}" --out "$S/smoke.jsonl" --shapes 256,2048 --warmup 0 --iters 0
  run validation "${base[@]}" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
  run memcheck compute-sanitizer --tool memcheck --error-exitcode 99 "${base[@]}" --out "$S/memcheck.jsonl" --warmup 0 --iters 0
  run synccheck compute-sanitizer --tool synccheck --error-exitcode 99 "${base[@]}" --out "$S/synccheck.jsonl" --warmup 0 --iters 0
  # Representative racecheck: both our paths (both kernels: tc_ladder runs tc_m32 at M >= 64), K=N=4096.
  run racecheck compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_tc \
    "${base[@]}" --out "$S/racecheck.jsonl" --shapes 4096,4096 --implementations tc tc_ladder --warmup 0 --iters 0
  "$PY" scripts/e050/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  (cd "$PRE" && sha256sum -c binaries.sha256 --ignore-missing) > "$S/prereq-binary-check.log"
  grep -q "$PRE/libe050_tc.so: OK" "$S/prereq-binary-check.log"
  cp "$PRE/binaries.sha256" "$S/prereq-binaries.sha256"
  base+=(--tc "$PRE/libe050_tc.so")
  clock > "$S/series1-clocks-before.csv"
  run series1 "${base[@]}" --out "$S/series1.jsonl"
  clock > "$S/series1-clocks-after.csv"
  run series2 "${base[@]}" --out "$S/series2.jsonl" --shapes 4096,4096 11008,4096 4096,11008 --ms 128 64 16 4 --series 2
  clock > "$S/series2-clocks-after.csv"
  "$PY" scripts/e050/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
