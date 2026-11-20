#!/usr/bin/env bash
# E058 inner run (one prun process on the node). inner.sh MODE SRC S LIB [PREREQ_DIR]
# Copy of E050's inner.sh: no library build (E050's libe050_tc.so, hash-checked), one shape, matrix counts 4/8/16.
set -euo pipefail
MODE=$1 SRC=$2 S=$3 LIB=$4 PRE=${5:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
# E010's runtime and preserved binaries (historical paths, identities checked below against E010's record).
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
E050SHA=a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 5400 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; free -g; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]

# Runtime: E010's pinned dependency set, restored without resolution only when absent (E005-R2 recipe, as E045/E050).
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
base=("$PY" scripts/e058/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")

if [[ $MODE == prereq ]]; then
  # E050's library, unchanged bytes: staged copy of results/e050_20261118/prereq-c8-343302/libe050_tc.so.
  cp "$LIB" "$S/libe050_tc.so"
  sha256sum "$S/libe050_tc.so" > "$S/tc-lib.sha256"
  [[ $(cut -d' ' -f1 "$S/tc-lib.sha256") == "$E050SHA" ]]
  grep -q "^$E050SHA .*prereq-c8-343302/libe050_tc.so$" results/e050_20261118/prereq-c8-343302/binaries.sha256
  E050LIB=/mnt/beegfs/hpuvvad/csc548-w4a16-e050-20261118/prereq-c8-343302/libe050_tc.so
  { [[ -e $E050LIB ]] && sha256sum "$E050LIB" || echo "absent: $E050LIB"; } > "$S/tc-lib-beegfs.txt"
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run project-gpu bash scripts/check_inblock.sh
  sha256sum "$S/libe050_tc.so" build/w4a16_bench src/cuda/*.cu src/cuda/*.cuh scripts/e058/* > "$S/binaries.sha256"
  base+=(--tc "$S/libe050_tc.so")
  run smoke "${base[@]}" --out "$S/smoke.jsonl" --shape 256,2048 --warmup 0 --iters 0
  run validation "${base[@]}" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
  clock > "$S/smoke-clocks-before.csv"
  run smoke-timing "${base[@]}" --out "$S/smoke-timing.jsonl" --ms 4
  clock > "$S/smoke-clocks-after.csv"
  run memcheck compute-sanitizer --tool memcheck --error-exitcode 99 "${base[@]}" --out "$S/memcheck.jsonl" --counts 16 --warmup 0 --iters 0
  run synccheck compute-sanitizer --tool synccheck --error-exitcode 99 "${base[@]}" --out "$S/synccheck.jsonl" --counts 16 --warmup 0 --iters 0
  "$PY" scripts/e058/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  (cd "$PRE" && sha256sum -c tc-lib.sha256) > "$S/prereq-binary-check.log"
  [[ $(sha256sum "$PRE/libe050_tc.so" | cut -d' ' -f1) == "$E050SHA" ]]
  cp "$PRE/binaries.sha256" "$S/prereq-binaries.sha256"
  base+=(--tc "$PRE/libe050_tc.so")
  clock > "$S/series1-clocks-before.csv"
  run series1 "${base[@]}" --out "$S/series1.jsonl"
  clock > "$S/series1-clocks-after.csv"
  run series2 "${base[@]}" --out "$S/series2.jsonl" --counts 16 8 4 --ms 128 64 16 4 --series 2
  clock > "$S/series2-clocks-after.csv"
  "$PY" scripts/e058/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
