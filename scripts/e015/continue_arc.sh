#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e015-continue
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodelist=c9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
set -euo pipefail
: "${PROJECT_ROOT:?Absolute BeeGFS root}" "${MODE:?storage, validation, primary or heldout}"
old=/home/hpuvvad/csc548-w4a16-e015-c7-20261109
if [[ "$MODE" == storage ]]; then
 [[ "$PROJECT_ROOT" == /mnt/beegfs/hpuvvad/csc548-w4a16-e015-c9-20261109 ]]
 if [[ ! -d /mnt/beegfs/hpuvvad ]]; then mkdir /mnt/beegfs/hpuvvad; chmod 700 /mnt/beegfs/hpuvvad; fi
 mkdir "$PROJECT_ROOT"
 cp -a "$old/scripts" "$old/src" "$old/include" "$old/tests" "$old/CMakeLists.txt" "$PROJECT_ROOT/"
 mkdir -p "$PROJECT_ROOT/results/e015_20261109"
 cp "$0" "$PROJECT_ROOT/scripts/e015/continue_arc.sh"
 cat > "$PROJECT_ROOT/scripts/e015/env_arc.sh" <<'E015_ENV'
#!/usr/bin/env bash
: "${PROJECT_ROOT:?Absolute E015 project root}" "${SLURM_JOB_ID:?Live allocation}"
[[ "$PROJECT_ROOT" == /* && -f "$PROJECT_ROOT/scripts/e015/run.py" ]]
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export E015_PYTHON=/tmp/e015-hpuvvad-20261109/bin/python
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export R="$PROJECT_ROOT/results/e015_20261109"
allocation() {
 scontrol show job "$SLURM_JOB_ID" > "$R/$1-allocation.txt"
 grep -q 'UserId=hpuvvad' "$R/$1-allocation.txt"
 grep -q 'JobState=RUNNING' "$R/$1-allocation.txt"
 grep -q "NodeList=$SLURM_JOB_NODELIST" "$R/$1-allocation.txt"
 [[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
 [[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
 [[ $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) == '12.0' ]]
}
run() {
 local label=$1
 shift
 [[ ! -e "$R/$label.log" && ! -e "$R/$label.exit" ]]
 local rc=0
 (set -x; prun timeout 1800 "$@") > "$R/$label.log" 2>&1 || rc=$?
 echo "$rc" > "$R/$label.exit"
 [[ $rc == 0 ]]
}
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
E015_ENV
 cp -a "$old/results/e015_20261109" "$PROJECT_ROOT/attempt1"
 for name in model.json e014-B-128.json e014-B-2048.json pip-freeze.txt dependency-diff.txt; do
  cp "$old/results/e015_20261109/$name" "$PROJECT_ROOT/results/e015_20261109/$name"
 done
 cp "$old/validation-slurm-331846.log" "$old/validation-slurm-331849.log" "$PROJECT_ROOT/attempt1/"
fi
source "$PROJECT_ROOT/scripts/e015/env_arc.sh"
exec > "$R/$MODE-driver.log" 2>&1
allocation "$MODE"
findmnt -T "$PROJECT_ROOT" > "$R/$MODE-storage.txt"
df -h "$PROJECT_ROOT" >> "$R/$MODE-storage.txt"
if [[ "$MODE" == storage ]]; then
 prun /bin/hostname > "$R/launcher.log" 2>&1
 [[ $(grep -c "^$SLURM_JOB_NODELIST$" "$R/launcher.log") == 1 ]]
 {
  date --iso-8601=seconds
  lscpu; lscpu -e; cat /proc/cpuinfo; numactl --hardware; cat /proc/self/status
  gcc --version; mpirun --version; nvcc --version; compute-sanitizer --version
  nvidia-smi -q; nvidia-smi topo -m
  env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
 } > "$R/metadata.txt" 2>&1
 export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
 if [[ ! -x "$E015_PYTHON" ]]; then
  run venv python3.11 -m venv /tmp/e015-hpuvvad-20261109
  run install "$E015_PYTHON" -m pip install -r scripts/e014/requirements.lock
 fi
 "$E015_PYTHON" -m pip freeze > "$R/current-pip-freeze.txt"
 diff -u "$R/pip-freeze.txt" "$R/current-pip-freeze.txt" > "$R/current-dependency-diff.txt"
 [[ $(df --output=avail -B1 "$PROJECT_ROOT" | tail -1) -gt 2147483648 ]]
 dd if=/dev/zero of="$R/storage-probe.bin" bs=1048576 count=1
 [[ $(stat -c %s "$R/storage-probe.bin") == 1048576 ]]
 sha256sum "$R/storage-probe.bin" > "$R/storage-probe.sha256"
 for variant in E G; do
  run "smoke_${variant}_128" "$E015_PYTHON" scripts/e015/run.py "$R" --stage smoke --variant "$variant" --context 128
 done
 touch "$R/SMOKE_CAPTURED"
elif [[ "$MODE" == validation ]]; then
 [[ -e "$R/SMOKE_INSPECTED" ]]
 run project-build bash scripts/build.sh
 run project-ctest ctest --test-dir build --output-on-failure
 run project-gpu-check bash scripts/check_inblock.sh
 sha256sum scripts/e014/{model.py,fetch.py,prompts.json,validate.py,timeline.py} scripts/e015/*.* > "$R/validation-tooling.sha256"
 for context in 128 2048; do
  for variant in E G; do
   run "validation_${variant}_${context}" "$E015_PYTHON" scripts/e015/run.py "$R" --stage validation --variant "$variant" --context "$context"
  done
 done
 run validation-analysis "$E015_PYTHON" scripts/e015/analyze.py "$R" --stage validation
elif [[ "$MODE" == primary || "$MODE" == heldout ]]; then
 export PHASE="$MODE"
 bash scripts/e015/measure_arc.sh
else
 exit 2
fi
clock > "$R/$MODE-clocks.csv"
touch "$R/${MODE^^}_CAPTURED"
# BeeGFS is compute-node-only; retain allocation briefly for inspected retrieval.
for ((attempt=0; attempt<300; attempt++)); do
 if [[ -e "$R/${MODE^^}_RETRIEVED" ]]; then exit 0; fi
 sleep 2
done
exit 124
