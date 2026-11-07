#!/usr/bin/env bash
# Invoked over SSH on c10 inside the existing interactive allocation.
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
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
 hostname
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
(cd "$E006_CONTROL"; head -n 3 results/e006_20261106/final-binaries.sha256 | sha256sum -c -) > "$R/e006-preserved-binaries-check.log"
# The fourth historical manifest entry was node-local. Verify its restored hash in identity.py.
bash scripts/e005_r2/restore_arc.sh
