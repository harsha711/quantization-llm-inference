#!/usr/bin/env bash
# Local helper: emits an E027 batch script (tooling embedded). The job runs on whichever node Slurm gives it.
#   scripts/e027/submit.sh smoke | ssh arc-off-network sbatch --parsable
#   scripts/e027/submit.sh full SMOKEDIR "128 512 2048" | ssh arc-off-network sbatch --parsable --nodelist=<smoke node>
set -euo pipefail
stage=$1 prior=${2:-none} contexts=${3:-128}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e027-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --output=/tmp/e027-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e027-20261110
VENV=/tmp/e017-hpuvvad-20261109
VV=/tmp/e027-vllm-hpuvvad-20261110
mkdir -p "\$OUT"; S="\$OUT/$stage-\$(hostname)-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
EOF
cat <<'EOF'
# Same runtime bytes as c7, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/../e021/validate.py"; embed "$here/../e023/run.py"; embed "$here/../e026/att.py"
embed "$here/convert.py"; embed "$here/vbench.py"; embed "$here/analyze.py"
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(hostname) == "$SLURM_JOB_NODELIST" ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; lscpu; nvidia-smi -q; env | grep -E '^(SLURM_|OMP_|OMPI_)' | sort; } > "$S/metadata.txt" 2>&1
sha256sum -c "$R/binaries.sha256" > "$S/binary-check.log"
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
# vLLM in its own venv: the pinned, hashed v0.33.0 wheel already used for the Marlin source, plus its PyPI dependencies.
export PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CONFIG_ROOT="$S/vllm-config" VLLM_CACHE_ROOT="$S/vllm-cache"
WHEEL=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-20261109/external/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
echo "4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa  $WHEEL" | sha256sum -c
if [[ ! -e "$VV/bin/python" ]]; then
 python3.11 -m venv "$VV"
 if [[ -e "$OUT/vllm-freeze.txt" ]]; then "$VV/bin/python" -m pip install "$WHEEL" -c "$OUT/vllm-freeze.txt" > "$S/vllm-install.log" 2>&1
 else "$VV/bin/python" -m pip install "$WHEEL" > "$S/vllm-install.log" 2>&1; fi
fi
"$VV/bin/python" -m pip freeze > "$S/vllm-freeze.txt"
[[ -e "$OUT/vllm-freeze.txt" ]] || cp "$S/vllm-freeze.txt" "$OUT/vllm-freeze.txt"
diff -u "$OUT/vllm-freeze.txt" "$S/vllm-freeze.txt"
go() { local label=$1; shift; (set -x; prun timeout 3000 "$E017_PYTHON" "$S/att.py" "$@") > "$S/$label.log" 2>&1; }
gov() { local label=$1; shift; (set -x; prun timeout 3000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq" "$S/ref" "$S/out" "$@") > "$S/$label.log" 2>&1; }
if [[ ! -e "$OUT/model-gptq.sha256" ]]; then
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/convert.py" "$OUT/model-gptq") > "$S/convert.log" 2>&1
 (cd "$OUT" && sha256sum model-gptq/* > model-gptq.sha256)
fi
(cd "$OUT" && sha256sum -c model-gptq.sha256) > "$S/model-check.log"
EOF
case $stage in
smoke) cat <<'EOF'
go validate validate "$S/ref" 128
mkdir "$S/out"; go time-GQA time "$S/out" GQA 128 heldout 1 "$S/ref"
gov time-V 128 heldout 1
EOF
;;
full) cat <<EOF
ls "\$OUT"/$prior/INSPECTED
contexts=($contexts)
EOF
cat <<'EOF'
go validate validate "$S/ref" "${contexts[@]}"
mkdir "$S/out"
for series in 1 2; do
 cs=("${contexts[@]}"); ps=(primary heldout); vs=(GR GQ GQA V)
 if [[ $series == 2 ]]; then cs=($(printf '%s\n' "${contexts[@]}" | tac)); ps=(heldout primary); vs=(V GQA GQ GR); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do
  label="${p}_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  if [[ $v == V ]]; then gov "$label" "$c" "$p" "$series"; else go "$label" time "$S/out" "$v" "$c" "$p" "$series" "$S/ref"; fi
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done; done
done
python3 "$S/analyze.py" "$S/out" "${contexts[@]}" > "$S/analysis.json"
EOF
;;
*) exit 2;;
esac
