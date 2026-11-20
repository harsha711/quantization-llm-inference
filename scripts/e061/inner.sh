#!/usr/bin/env bash
# E061 inner run (one prun process on the node). inner.sh SRC S
# Copy of E058's inner.sh: no library build, no Marlin/tc; one job = checks, tests, validation, two series.
set -euo pipefail
SRC=$1 S=$2
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
LOCAL=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 3600 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; free -g; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]

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
sha256sum scripts/e061/* > "$S/e061-scripts.sha256"

# Unit tests and mutation check first (hand-derived expectations); nothing is timed unless both pass.
run unit-tests "$PY" -m unittest -v scripts/e061/test_e061.py
run mutations "$PY" scripts/e061/mutate.py
run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
run project-ctest ctest --test-dir build --output-on-failure
run project-gpu bash scripts/check_inblock.sh
base=("$PY" scripts/e061/bench.py --source "$SOURCE" --local "$LOCAL" --wheel "$WHEEL" --bridge "$BRIDGE")
run validation "${base[@]}" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
clock > "$S/series1-clocks-before.csv"
run series1 "${base[@]}" --out "$S/series1.jsonl" --series 1
clock > "$S/series1-clocks-after.csv"
run series2 "${base[@]}" --out "$S/series2.jsonl" --series 2
clock > "$S/series2-clocks-after.csv"
# Exit 1 = some cell not floorcheck OK (recorded in analysis.exit); raw series are kept either way.
rc=0; "$PY" scripts/e061/analyze.py "$S" > "$S/summary.json" 2> "$S/analysis.txt" || rc=$?; echo "$rc" > "$S/analysis.exit"
[[ $rc == 0 ]]
echo INNER-OK
