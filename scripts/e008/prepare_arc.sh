#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e008/env_arc.sh"
mkdir -p "$PROJECT_ROOT/results"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
grep -q 'UserId=hpuvvad' "$R/allocation.txt"
grep -q 'JobState=RUNNING' "$R/allocation.txt"
grep -q 'NodeList=c10' "$R/allocation.txt"
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c10$' "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 lscpu
 lscpu -e
 cat /proc/cpuinfo
 numactl --hardware
 cat /proc/self/status
 gcc --version
 mpirun --version
 nvcc --version
 compute-sanitizer --version
 ncu --version
 nvidia-smi -q
 nvidia-smi topo -m
 findmnt -T "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|LD_LIBRARY_PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/handoff.sha256) > "$R/e006-handoff-check.log"
(cd /home/hpuvvad/csc548-w4a16-e005-r2-20261106; sha256sum -c results/e005_r2_c10_20261106/handoff.sha256) > "$R/e005-r2-handoff-check.log"
cycle_base
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E004_CONTROL/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
sha256sum "$E005_LOCAL" "$E005_BRIDGE" "$E005_WHEEL" scripts/e005/bench.py scripts/e004/bench.py > "$R/verified-inputs.sha256"
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run identity "$E005_PYTHON" scripts/e005_r2/identity.py
run smoke "${base[@]}" --out "$R/smoke.jsonl" --shapes 256,2048 --warmup 0 --iters 0
touch "$R/SMOKE_PASSED"
