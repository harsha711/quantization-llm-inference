#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r1/env_arc.sh"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
grep -q 'UserId=hpuvvad' "$R/allocation.txt"
grep -q 'JobState=RUNNING' "$R/allocation.txt"
grep -q 'NodeList=c58' "$R/allocation.txt"
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/handoff.sha256) > "$R/e006-handoff-check.log"
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/final-binaries.sha256) > "$R/e006-binaries-check.log"
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c58$' "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds
 hostname
 lscpu
 numactl --hardware
 cat /proc/self/status
 gcc --version
 mpirun --version
 nvcc --version
 compute-sanitizer --version
 ncu --version
 nvidia-smi -q
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|LD_LIBRARY_PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u /home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/pip-freeze.txt "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run identity "$E005_PYTHON" scripts/e005_r1/identity.py
run smoke "${base[@]}" --out "$R/smoke.jsonl" --ms 1 5 8 --shapes 256,2048 --patterns mixed zero_a zero_w \
 --warmup 0 --iters 0 --policies no_flush --repeats 3
sha256sum "$E005_LOCAL" "$E005_BRIDGE" "$E005_WHEEL" scripts/e005/bench.py scripts/e004/bench.py > "$R/verified-inputs.sha256"
sha256sum scripts/e005_r1/* scripts/e005/bench.py scripts/e004/bench.py > "$R/prerequisite-tooling.sha256"
# A separate R1 gate certifies identity and smoke completion, never historical E005.
touch "$R/PREREQUISITE_PASSED"
