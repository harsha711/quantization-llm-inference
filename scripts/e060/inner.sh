#!/usr/bin/env bash
# E060 inner run (one prun process on the node). inner.sh MODE SRC S [PREREQ_DIR]
# Copy of E058's inner.sh: E006's matched control/patched libe005.so (no build), unit tests + mutation check first.
set -euo pipefail
MODE=$1 SRC=$2 S=$3 PRE=${4:-}
cd "$SRC"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OMP_PROC_BIND=true
# E010's runtime and preserved binaries (historical paths, identities checked below).
VENV=/tmp/e005-r2-hpuvvad-20261106 PY=/tmp/e005-r2-hpuvvad-20261106/bin/python
E004=/home/hpuvvad/csc548-w4a16-e004-20261106
E006R=/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106
BRIDGE=$E004/results/e004_20261106/libe004.so
SOURCE=/home/hpuvvad/csc548-w4a16-e005-20261106/external/pinned
CTL=51cfcf3a8c3758631e2824d0be8f0da4c6a4cf29021a769ccf864afb4d1d652a
PAT=e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv; }
run() { local label=$1 rc=0; shift; (set -x; timeout 5400 "$@") > "$S/$label.log" 2>&1 || rc=$?; echo "$rc" > "$S/$label.exit"; [[ $rc == 0 ]]; }
{ date -Is; hostname; nproc; lscpu; numactl --hardware; free -g; cat /proc/self/status; nvidia-smi -q; nvcc --version; c++ --version; cmake --version
  compute-sanitizer --version; env | sort | grep -E '^(SLURM|OMP|OMPI)'; } > "$S/metadata.txt" 2>&1
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]

# Runtime: E010's pinned dependency set, restored without resolution only when absent (as E045/E050/E058).
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
# E006's matched binaries, unchanged bytes (no rebuild): both hashes, and the same values in E006's own manifest.
sha256sum "$E006R/control/libe005.so" "$E006R/patched/libe005.so" > "$S/libraries.sha256"
grep -q "^$CTL  $E006R/control/libe005.so$" "$S/libraries.sha256"
grep -q "^$PAT  $E006R/patched/libe005.so$" "$S/libraries.sha256"
grep -q "^$CTL  results/e006_20261106/control/libe005.so$" results/e006_20261106/final-binaries.sha256
grep -q "^$PAT  results/e006_20261106/patched/libe005.so$" results/e006_20261106/final-binaries.sha256
sha256sum scripts/e060/* scripts/checks/floorcheck.py > "$S/scripts.sha256"
# User's test-driven rule: unit tests (hand-derived expectations) and the mutation check gate every job.
run unit-tests "$PY" -m unittest -v scripts/e060/test_e060.py
run mutation "$PY" scripts/e060/mutate.py
base=("$PY" scripts/e060/bench.py --source "$SOURCE" --local "$E006R/patched/libe005.so" --wheel "$WHEEL"
      --bridge "$BRIDGE" --e006-root "$E006R")

if [[ $MODE == prereq ]]; then
  run project-build env BUILD_DIR="$SRC/build" bash scripts/build.sh
  run project-ctest ctest --test-dir build --output-on-failure
  run project-gpu bash scripts/check_inblock.sh
  run smoke "${base[@]}" --out "$S/smoke.jsonl" --shapes 256,2048 --matrices 2 --warmup 0 --iters 0
  run validation "${base[@]}" --out "$S/validation.jsonl" --trace --warmup 0 --iters 0
  clock > "$S/smoke-clocks-before.csv"
  run smoke-timing "${base[@]}" --out "$S/smoke-timing.jsonl" --shapes 4096,11008 --ms 1 4
  clock > "$S/smoke-clocks-after.csv"
  san=(compute-sanitizer --error-exitcode 99 --kernel-name kns=e005_marlin)
  run memcheck "${san[@]}" --tool memcheck "${base[@]}" --out "$S/memcheck.jsonl" --arms patched --warmup 0 --iters 0
  run synccheck "${san[@]}" --tool synccheck "${base[@]}" --out "$S/synccheck.jsonl" --arms patched --warmup 0 --iters 0
  # E006's minimized racecheck case (256 x 2048, M8, one matrix), concise analysis report, each arm separately.
  rc=(--tool racecheck --racecheck-report analysis)
  small=(--shapes 256,2048 --matrices 1 --ms 8 --warmup 0 --iters 0)
  run racecheck-patched "${san[@]}" "${rc[@]}" "${base[@]}" --out "$S/racecheck-patched.jsonl" --arms patched "${small[@]}"
  # The control body's known hazard is expected here: recorded (exit code and summary), not suppressed, not a failure.
  run racecheck-control "${san[@]}" "${rc[@]}" "${base[@]}" --out "$S/racecheck-control.jsonl" --arms control "${small[@]}" || true
  "$PY" scripts/e060/analyze.py "$S" --stage prereq > "$S/prereq-check.log"
else
  [[ -e "$PRE/INSPECTED" ]]
  grep -q '^PREREQ PASS' "$PRE/prereq-check.log"
  # Same GPU and driver as the inspected prerequisite (its metadata.txt), before any timing (as E058/E059).
  want_uuid=$(awk -F': ' '/^ *GPU UUID/{print $2; exit}' "$PRE/metadata.txt")
  want_drv=$(awk -F': ' '/^Driver Version/{split($2,a," "); print a[1]; exit}' "$PRE/metadata.txt")
  read -r got_uuid got_drv < <(nvidia-smi --query-gpu=uuid,driver_version --format=csv,noheader | tr -d ',')
  echo "prereq $want_uuid $want_drv / now $got_uuid $got_drv" > "$S/gpu-identity.txt"
  [[ -n $want_uuid && -n $want_drv && $got_uuid == "$want_uuid" && $got_drv == "$want_drv" ]]
  cmp "$PRE/libraries.sha256" "$S/libraries.sha256"
  clock > "$S/series1-clocks-before.csv"
  run series1 "${base[@]}" --out "$S/series1.jsonl"
  clock > "$S/series1-clocks-after.csv"
  run series2 "${base[@]}" --out "$S/series2.jsonl" --shapes 11008,4096 4096,11008 4096,4096 --ms 128 64 16 4 1 --series 2
  clock > "$S/series2-clocks-after.csv"
  "$PY" scripts/e060/analyze.py "$S" --stage full > "$S/analysis.txt"
fi
echo INNER-OK
