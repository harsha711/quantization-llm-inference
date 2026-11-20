#!/usr/bin/env bash
# Local helper: emits one E056 batch script (tooling embedded, as E052/E054). MODE: prereq | full.
#   scripts/e056/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=c10
#   scripts/e056/submit.sh full PREREQ_DIR "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=c10 --dependency=afterok:JOB
# E030's per-step kernel profile replayed with our tc_ladder: GTA through E052's unchanged t.py (exact E050 library), GQA through
# t.py --marlin (bridge unchanged), V = E030's unchanged vprofile.py (vLLM v0.33.0 in-process). Same node, same job.
set -euo pipefail
mode=$1 pre=${2:-} note=${3:-}
[[ $mode == prereq || ( $mode == full && -n $pre && -n $note ) ]]
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e056-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$(case $mode in prereq) echo 01:30:00;; *) echo 03:00:00;; esac)
#SBATCH --output=/tmp/e056-$mode-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
EOF
cat <<'EOF'
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e056-20261119
VENV=/tmp/e017-hpuvvad-20261109
VV=/tmp/e027-vllm-hpuvvad-20261110
E27=/mnt/beegfs/hpuvvad/csc548-w4a16-e027-20261110
export E052_LIB=/mnt/beegfs/hpuvvad/csc548-w4a16-e050-20261118/prereq-c8-343302/libe050_tc.so
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
# Copy the finished directory (no *.npz) to home for retrieval from the login node.
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";tar -C "$OUT" -czf "$HOME/e056-$(basename "$S").tgz" --exclude="*.npz" --exclude=vllm-cache "$(basename "$S")";sha256sum "$HOME/e056-$(basename "$S").tgz" > "$HOME/e056-$(basename "$S").tgz.sha256";exit "$rc"' EXIT
step() { echo "STEP $* $(date -Is)"; }
step metadata
# Same runtime bytes as every earlier model job, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then echo restored > "$S/runtime-restored.txt"; sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
for f in ../e018/diagnose.py ../e021/validate.py ../e023/run.py ../e026/att.py ../e030/gprofile.py ../e030/vprofile.py ../e052/t.py analyze.py; do embed "$here/$f"; done
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; lscpu; free -g; nvidia-smi -q; nvcc --version; env | grep -E '^(SLURM_|OMP_|OMPI_)' | sort; } > "$S/metadata.txt" 2>&1
sha256sum -c "$R/binaries.sha256" > "$S/binary-check.log"
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
echo "a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe  $E052_LIB" | sha256sum -c > "$S/lib-check.log"
{ ldd "$E052_LIB"; nm -D "$E052_LIB" | grep e050_; } > "$S/lib-info.txt" 2>&1
{ for d in "$VENV" "$VV"; do
  if [[ -e $d/INSTALLED || -x $d/bin/python ]]; then echo "PRESENT $d $(ls -d --full-time $d/INSTALLED 2>/dev/null)"; else echo "ABSENT $d"; fi; done; } > "$S/venv-inventory.txt"
if [[ -n $PRE_NAME ]]; then PRE="$OUT/$PRE_NAME"; [[ $(cat "$PRE/driver.exit") == 0 ]]; [[ -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"; fi
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
# t KERNEL LABEL SCRIPT ARGS: KERNEL T = tc_ladder (E052 t.py), M = corrected Marlin (bridge unchanged)
t() { local k=$1 label=$2 flag=; shift 2; [[ $k == M ]] && flag=--marlin
 (set -x; prun timeout 9000 "$E017_PYTHON" "$S/t.py" $flag "$@") > "$S/$label.log" 2>&1; }
# vLLM: E052's recipe (restored from the pinned hashed wheel with E027's freeze if absent; the freeze diff must be empty)
export PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CONFIG_ROOT="$S/vllm-config" VLLM_CACHE_ROOT="$S/vllm-cache"
WHEEL=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-20261109/external/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
echo "4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa  $WHEEL" | sha256sum -c
if [[ ! -e "$VV/INSTALLED" ]]; then
 echo restored > "$S/vllm-restored.txt"; rm -rf "$VV"; python3.11 -m venv "$VV"
 "$VV/bin/python" -m pip install "$WHEEL" -c "$E27/vllm-freeze.txt" > "$S/vllm-install.log" 2>&1; touch "$VV/INSTALLED"
fi
"$VV/bin/python" -m pip freeze > "$S/vllm-freeze.txt"; diff -u "$E27/vllm-freeze.txt" "$S/vllm-freeze.txt"
(cd "$E27" && sha256sum -c model-gptq.sha256) > "$S/vllm-model-check.log"
case $MODE in prereq) CONTEXTS=(128);; full) CONTEXTS=(128 2048);; *) exit 2;; esac
step gate
t T gate-graph att.py validate "$S/refT" "${CONTEXTS[@]}"
t M control-graph att.py validate "$S/ref" "${CONTEXTS[@]}"
grep -h "ACCEPTED\|REJECTED" "$S"/gate-*.log "$S"/control-*.log > "$S/gate-summary.txt"
! grep -q REJECTED "$S/gate-summary.txt"
step profile
clock > "$S/clk-before.csv"
t T prof-T gprofile.py "$S/prof-T" "${CONTEXTS[@]}"
clock > "$S/clk-after-T.csv"
t M prof-M gprofile.py "$S/prof-M" "${CONTEXTS[@]}"
clock > "$S/clk-after-M.csv"
mkdir "$S/prof-V"; cp "$S/prof-M/ours.json" "$S/prof-V/ours.json"   # vprofile's token reference = the Marlin run
(set -x; prun timeout 6000 "$VV/bin/python" "$S/vprofile.py" "$E27/model-gptq" "$S/prof-V" "${CONTEXTS[@]}") > "$S/prof-V.log" 2>&1
clock > "$S/clk-after-V.csv"
step analyze
python3 "$S/analyze.py" "$S" > "$S/analysis.json" 2> "$S/analysis.txt"
(cd "$S" && find . -type f ! -path './vllm-cache/*' ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 ! -name '*.npz' | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF
