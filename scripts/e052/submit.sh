#!/usr/bin/env bash
# Local helper: emits one E052 batch script (tooling embedded, as E026-E038). MODE: prereq | core | long | head | ext.
#   scripts/e052/submit.sh prereq | ssh arc-off-network sbatch --parsable --nodelist=cN
#   scripts/e052/submit.sh core PREREQ_DIR "inspection note" | ssh arc-off-network sbatch --parsable --nodelist=cN --dependency=afterok:JOB
# Our tc_ladder arms run through t.py (bridge patched); Marlin and FP16 arms through t.py --marlin (bridge unchanged).
set -euo pipefail
mode=$1 pre=${2:-} note=${3:-}
[[ $mode == prereq || ( -n $pre && -n $note ) ]]
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e052-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=$(case $mode in prereq) echo 02:00:00;; *) echo 10:00:00;; esac)
#SBATCH --output=/tmp/e052-$mode-%j.log
MODE=$mode PRE_NAME=$pre
NOTE="$note"
EOF
cat <<'EOF'
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e052-20261118
VENV=/tmp/e017-hpuvvad-20261109
VV=/tmp/e027-vllm-hpuvvad-20261110
E27=/mnt/beegfs/hpuvvad/csc548-w4a16-e027-20261110 E28=/mnt/beegfs/hpuvvad/csc548-w4a16-e028-20261110
X=/mnt/beegfs/hpuvvad/csc548-w4a16-e038-20261111
export E052_LIB=/mnt/beegfs/hpuvvad/csc548-w4a16-e050-20261118/prereq-c8-343302/libe050_tc.so
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
# Same runtime bytes as c7, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then echo restored > "$S/runtime-restored.txt"; sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
for f in ../e018/diagnose.py ../e021/validate.py ../e023/run.py ../e026/att.py ../e028/long.py ../e031/head.py ../e038/ckpt.py \
         ../e027/vbench.py ../e028/obench.py ../e028/lbench.py t.py opcheck.py analyze.py; do embed "$here/$f"; done
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; lscpu; nvidia-smi -q; nvcc --version; compute-sanitizer --version; env | grep -E '^(SLURM_|OMP_|OMPI_)' | sort; } > "$S/metadata.txt" 2>&1
sha256sum -c "$R/binaries.sha256" > "$S/binary-check.log"
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
echo "a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe  $E052_LIB" | sha256sum -c > "$S/lib-check.log"
{ ldd "$E052_LIB"; nm -D "$E052_LIB" | grep e050_; } > "$S/lib-info.txt" 2>&1
{ for d in /tmp/e017-hpuvvad-20261109 /tmp/e027-vllm-hpuvvad-20261110 /tmp/e028-ort-hpuvvad-20261110 /tmp/e028-llama-hpuvvad-20261110; do
  if [[ -e $d/INSTALLED || -x $d/bin/python ]]; then echo "PRESENT $d $(ls -d --full-time $d/INSTALLED 2>/dev/null)"; else echo "ABSENT $d"; fi; done; } > "$S/venv-inventory.txt"
PRE=; [[ -n $PRE_NAME ]] && PRE="$OUT/$PRE_NAME"
[[ -z $PRE || -e "$PRE/INSPECTED" ]] || echo "$NOTE" > "$PRE/INSPECTED"
[[ -z $PRE ]] || ls "$PRE/INSPECTED"
clock() { nvidia-smi --query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv; }
# t KERNEL LABEL SCRIPT ARGS: KERNEL T = tc_ladder, M = corrected Marlin (bridge unchanged)
t() { local k=$1 label=$2 flag=; shift 2; [[ $k == M ]] && flag=--marlin
 (set -x; prun timeout 9000 "$E017_PYTHON" "$S/t.py" $flag "$@") > "$S/$label.log" 2>&1; }
note() { printf '{"arm":"%s","context":%s,"prompt":"%s","series":%s,"file":"%s"}\n' "$1" "$2" "$3" "$4" "$5" >> "$S/cells.jsonl"; }
vllm_env() {
 export PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip
 export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CONFIG_ROOT="$S/vllm-config" VLLM_CACHE_ROOT="$S/vllm-cache"
 WHEEL=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-20261109/external/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
 echo "4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa  $WHEEL" | sha256sum -c
 if [[ ! -e "$VV/INSTALLED" ]]; then  # E027's recipe, constrained to its recorded freeze; the diff below must be empty
  echo restored > "$S/vllm-restored.txt"; rm -rf "$VV"; python3.11 -m venv "$VV"
  "$VV/bin/python" -m pip install "$WHEEL" -c "$E27/vllm-freeze.txt" > "$S/vllm-install.log" 2>&1; touch "$VV/INSTALLED"
 fi
 "$VV/bin/python" -m pip freeze > "$S/vllm-freeze.txt"; diff -u "$E27/vllm-freeze.txt" "$S/vllm-freeze.txt"
 (cd "$E27" && sha256sum -c model-gptq.sha256 model-gptq-head.sha256) > "$S/vllm-model-check.log"; }
# One timed cell. Arms: GR GRA GQ GQA (FP16/Marlin graph), GT GTA (tc_ladder graph), QA TA (eager, direct attention),
# GQAH GTAH (4-bit head), c* (E038 calibrated root), V VH cV cVH (vLLM), OF OQ (ORT), LF LQ (llama.cpp).
cell() { local arm=$1 c=$2 p=$3 s=$4 o; local label="${p}_${arm}_${c}_s${s}"
 clock > "$S/clk-$label.csv"
 case $arm in
 GR|GRA|GQ|GQA) o=out; t M "$label" $ATT time "$S/$o" "$arm" "$c" "$p" "$s" "$S/ref"; f="$S/$o/${p}_${arm}_${c}_s${s}.json";;
 GT|GTA) o=outT; v=${arm/T/Q}; t T "$label" $ATT time "$S/$o" "$v" "$c" "$p" "$s" "$S/refT"; f="$S/$o/${p}_${v}_${c}_s${s}.json";;
 QA) o=outE; t M "$label" --eager att.py time "$S/$o" GQA "$c" "$p" "$s" "$S/ref"; f="$S/$o/${p}_GQA_${c}_s${s}.json";;
 TA) o=outET; t T "$label" --eager att.py time "$S/$o" GQA "$c" "$p" "$s" "$S/refT"; f="$S/$o/${p}_GQA_${c}_s${s}.json";;
 GQAH|hGQA) v=${arm#h}; t M "$label" head.py time "$S/outH" "$v" "$c" "$p" "$s" "$S/refH"; f="$S/outH/${p}_${v}_${c}_s${s}.json";;
 GTAH|hGTA) v=${arm#h}; v=${v/T/Q}; t T "$label" head.py time "$S/outHT" "$v" "$c" "$p" "$s" "$S/refHT"; f="$S/outHT/${p}_${v}_${c}_s${s}.json";;
 cGQA|cGQAH) v=${arm#c}; E017_ROOT=$X/root t M "$label" ckpt.py head time "$S/outC" "$v" "$c" "$p" "$s" "$S/refC"; f="$S/outC/${p}_${v}_${c}_s${s}.json";;
 cGTA|cGTAH) v=${arm#c}; v=${v/T/Q}; E017_ROOT=$X/root t T "$label" ckpt.py head time "$S/outCT" "$v" "$c" "$p" "$s" "$S/refCT"; f="$S/outCT/${p}_${v}_${c}_s${s}.json";;
 V|VH|cV|cVH) local m=$E27/model-gptq r=$S/refH key=GQA d=outV
   case $arm in VH) m=$E27/model-gptq-head key=GQAH;; cV) m=$X/model-gptq-c r=$S/refC d=outVC;; cVH) m=$X/model-gptq-c-head r=$S/refC key=GQAH d=outVC;; esac
   [[ $MODE == ext ]] && r=$S/ref
   mkdir -p "$S/$d"; (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$m" "$r" "$S/$d" "$c" "$p" "$s" "${arm#c}" "$key") > "$S/$label.log" 2>&1
   f="$S/$d/${p}_${arm#c}_${c}_s${s}.json";;
 OF|OQ) local m=$E28/ort/fp16; [[ $arm == OQ ]] && m=$E28/ort/int4
   (set -x; prun timeout 6000 /tmp/e028-ort-hpuvvad-20261110/bin/python "$S/obench.py" "$m" "$S/ref" "$S/outX" "$arm" "$c" "$p" "$s") > "$S/$label.log" 2>&1; f="$S/outX/${p}_${arm}_${c}_s${s}.json";;
 LF|LQ) local m=$E28/gguf/qwen-f16.gguf; [[ $arm == LQ ]] && m=$E28/gguf/qwen-q4_k_m.gguf
   (set -x; prun timeout 6000 /tmp/e028-llama-hpuvvad-20261110/bin/python "$S/lbench.py" "$m" "$S/ref" "$S/outX" "$arm" "$c" "$p" "$s") > "$S/$label.log" 2>&1; f="$S/outX/${p}_${arm}_${c}_s${s}.json";;
 esac
 note "$arm" "$c" "$p" "$s" "$f"; ls "$f" > /dev/null; }
# Two fully reversed series over contexts x passages x arms.
series() { local cs=("${CONTEXTS[@]}") ps=(primary heldout) vs=("${ARMS[@]}")
 for s in 1 2; do
  if [[ $s == 2 ]]; then cs=($(printf '%s\n' "${CONTEXTS[@]}" | tac)); ps=(heldout primary); vs=($(printf '%s\n' "${ARMS[@]}" | tac)); fi
  for c in "${cs[@]}"; do for p in "${ps[@]}"; do for a in "${vs[@]}"; do cell "$a" "$c" "$p" "$s"; done; done; done
 done; }
mkdir -p "$S/out" "$S/outT" "$S/outE" "$S/outET" "$S/outH" "$S/outHT" "$S/outC" "$S/outCT" "$S/outX"
ATT=att.py
case $MODE in
prereq)
 t T opcheck opcheck.py "$S/opcheck"
 (set -x; prun timeout 9000 compute-sanitizer --tool memcheck --error-exitcode 99 "$E017_PYTHON" "$S/opcheck.py" "$S/memcheck" --quick) > "$S/memcheck.log" 2>&1
 (set -x; prun timeout 9000 compute-sanitizer --tool racecheck --error-exitcode 99 --kernel-name kns=w4a16_tc "$E017_PYTHON" "$S/opcheck.py" "$S/racecheck" --quick) > "$S/racecheck.log" 2>&1
 t T gate-eager-128 validate.py "$S/gateE" 64 128          # E021 gate, eager tc_ladder, 196 x 64 same-input operator checks
 t T gate-graph-128 att.py validate "$S/refT" 128           # E026 gate, graph + direct attention
 t M control-graph-128 att.py validate "$S/ref" 128
 cell GTA 128 heldout 1; cell GQA 128 heldout 1
 grep -h "ACCEPTED\|REJECTED" "$S"/gate-*.log "$S"/control-*.log > "$S/gate-summary.txt"
 ;;
core)
 for c in 128 256 512 1024 2048; do t T "gate-eager-$c" validate.py "$S/gateE-$c" 64 "$c"; done
 t T gate-graph att.py validate "$S/refT" 128 256 512 1024 2048
 t M control-graph att.py validate "$S/ref" 128 2048
 CONTEXTS=(128 2048) ARMS=(GR GRA GQ GQA GT GTA QA TA); series
 ;;
long)
 ATT=long.py
 t T gate-long long.py validate "$S/refT" 4096 8192 16384
 t M control-long long.py validate "$S/ref" 4096 8192 16384
 CONTEXTS=(4096 8192 16384) ARMS=(GRA GQA GTA); series
 ;;
head)
 export E038_CKPT=$(ls /mnt/beegfs/hpuvvad/csc548-w4a16-e037-20261111/calib-c4-335301/checkpoint.npz)
 echo "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_sha256'])" "$(dirname "$E038_CKPT")/quality.json")  $E038_CKPT" | sha256sum -c
 (cd "$X" && sha256sum -c built.sha256) > "$S/built-check.log"
 vllm_env
 t T gate-head head.py validate "$S/refHT" 128 2048
 t M control-head head.py validate "$S/refH" 128 2048
 E017_ROOT=$X/root t T gate-head-c ckpt.py head validate "$S/refCT" 128 2048
 E017_ROOT=$X/root t M control-head-c ckpt.py head validate "$S/refC" 128 2048
 CONTEXTS=(128 2048) ARMS=(hGQA GQAH hGTA GTAH V VH cGQA cGQAH cGTA cGTAH cV cVH); series
 sed -i 's/"arm":"hG/"arm":"G/' "$S/cells.jsonl"
 ;;
ext)
 vllm_env
 ARMS=(GQA GTA V)
 if [[ -e /tmp/e028-ort-hpuvvad-20261110/INSTALLED ]]; then
  /tmp/e028-ort-hpuvvad-20261110/bin/python -m pip freeze > "$S/ort-freeze.txt"; diff -u "$E28/ort-freeze.txt" "$S/ort-freeze.txt"
  (cd "$E28" && sha256sum -c ort.sha256) > "$S/ort-model-check.log"; ARMS+=(OF OQ)
 else echo "ORT UNAVAILABLE on $(hostname)" >> "$S/unavailable.txt"; fi
 if [[ -e /tmp/e028-llama-hpuvvad-20261110/INSTALLED ]]; then
  /tmp/e028-llama-hpuvvad-20261110/bin/python -m pip freeze > "$S/llama-freeze.txt"; diff -u "$E28/llama-freeze.txt" "$S/llama-freeze.txt"
  (cd "$E28" && sha256sum -c gguf.sha256) > "$S/llama-model-check.log"; ARMS+=(LF LQ)
 else echo "llama.cpp UNAVAILABLE on $(hostname)" >> "$S/unavailable.txt"; fi
 t T gate-graph att.py validate "$S/refT" 128 512 2048
 t M control-graph att.py validate "$S/ref" 128 512 2048
 CONTEXTS=(128 512 2048); series
 ;;
*) exit 2;;
esac
[[ $MODE == prereq ]] || python3 "$S/analyze.py" "$S" "$MODE" > "$S/analysis.json"
(cd "$S" && find . -type f ! -name FINISHED ! -name RETRIEVED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF
