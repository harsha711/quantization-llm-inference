#!/usr/bin/env bash
# Local helper: emits one batch script (tooling embedded). KIND: llama | ort (E028 external stacks) | long (E029 long contexts).
#   scripts/e028/submit.sh llama smoke | ssh arc-off-network sbatch --parsable
#   scripts/e028/submit.sh llama full SMOKEDIR "128 512 2048" | ssh arc-off-network sbatch --parsable --nodelist=<smoke node>
#   scripts/e028/submit.sh long smoke none 4096 | ...       scripts/e028/submit.sh long full SMOKEDIR "4096 8192 16384" | ...
set -euo pipefail
kind=$1 stage=$2 prior=${3:-none} contexts=${4:-128}
exp=e028; [[ $kind == long ]] && exp=e029
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-$exp-$kind-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/tmp/$exp-$kind-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-$exp-20261110
VENV=/tmp/e017-hpuvvad-20261109
XV=/tmp/e028-$kind-hpuvvad-20261110
mkdir -p "\$OUT"; S="\$OUT/$kind-$stage-\$(hostname)-\$SLURM_JOB_ID\${E_TAG:-}"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
EOF
cat <<'EOF'
# Same runtime bytes as c7, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/../e021/validate.py"; embed "$here/../e023/run.py"; embed "$here/../e026/att.py"
case $kind in
llama) embed "$here/lbench.py"; embed "$here/lquant.py"; embed "$here/analyze.py";;
ort) embed "$here/obench.py"; embed "$here/analyze.py";;
long) embed "$here/long.py"; embed "$here/../e026/analyze.py";;
esac
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
export PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip
ATT=att.py
MODEL=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["path"])' "$R/model.json")
go() { local label=$1; shift; (set -x; prun timeout 6000 "$E017_PYTHON" "$S/$ATT" "$@") > "$S/$label.log" 2>&1; }
freeze() { "$XV/bin/python" -m pip freeze > "$S/ext-freeze.txt"; [[ -e "$OUT/$1-freeze.txt" ]] || cp "$S/ext-freeze.txt" "$OUT/$1-freeze.txt"; diff -u "$OUT/$1-freeze.txt" "$S/ext-freeze.txt"; }
constraint() { [[ -e "$OUT/$1-freeze.txt" ]] && echo "-c $OUT/$1-freeze.txt" || true; }
EOF
case $kind in
llama) cat <<'EOF'
# llama.cpp through llama-cpp-python built from its sdist with CUDA; the conversion script comes from the same sdist.
if [[ ! -e "$OUT/llama-sdist.sha256" ]]; then
 url=$(curl -fsSL https://pypi.org/pypi/llama-cpp-python/json | python3 -c 'import json,sys;print(next(u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"))')
 mkdir "$OUT/llama-sdist"; (cd "$OUT/llama-sdist" && curl -fsSLO "$url"); (cd "$OUT" && sha256sum llama-sdist/* > llama-sdist.sha256)
fi
(cd "$OUT" && sha256sum -c llama-sdist.sha256)
if [[ ! -e "$XV/INSTALLED" ]]; then
 rm -rf "$XV"; python3.11 -m venv "$XV"
 CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF" CMAKE_BUILD_PARALLEL_LEVEL=16 "$XV/bin/python" -m pip install -v "$OUT"/llama-sdist/*.tar.gz torch transformers sentencepiece safetensors protobuf $(constraint llama) > "$S/ext-install.log" 2>&1
 touch "$XV/INSTALLED"
fi
freeze llama
F="$OUT/gguf/qwen-f16.gguf" Q="$OUT/gguf/qwen-q4_k_m.gguf"
if [[ ! -e "$OUT/gguf.sha256" ]]; then
 mkdir "$S/src" "$OUT/gguf"; tar xzf "$OUT"/llama-sdist/*.tar.gz -C "$S/src" --strip-components=1
 (set -x; "$XV/bin/python" "$S/src/vendor/llama.cpp/convert_hf_to_gguf.py" "$MODEL" --outfile "$F" --outtype f16) > "$S/convert.log" 2>&1
 (set -x; "$XV/bin/python" "$S/lquant.py" "$F" "$Q") > "$S/quantize.log" 2>&1
 rm -rf "$S/src"; (cd "$OUT" && sha256sum gguf/* > gguf.sha256)
fi
(cd "$OUT" && sha256sum -c gguf.sha256) > "$S/model-check.log"
gox() { local label=$1 v=$2; shift 2; local m=$F; [[ $v == LQ ]] && m=$Q; (set -x; prun timeout 6000 "$XV/bin/python" "$S/lbench.py" "$m" "$S/ref" "$S/out" "$v" "$@") > "$S/$label.log" 2>&1; }
EXT=(LF LQ)
EOF
;;
ort) cat <<'EOF'
# ONNX Runtime GenAI (CUDA provider) from PyPI; its model builder exports FP16 and its own INT4 from the same base model.
if [[ ! -e "$XV/INSTALLED" ]]; then
 rm -rf "$XV"; python3.11 -m venv "$XV"
 "$XV/bin/python" -m pip install onnxruntime-genai-cuda onnxruntime-gpu onnx onnx-ir torch transformers safetensors $(constraint ort) > "$S/ext-install.log" 2>&1
 touch "$XV/INSTALLED"
fi
freeze ort
if [[ ! -e "$OUT/ort.sha256" ]]; then
 rm -rf "$OUT/ort"; mkdir "$OUT/ort"
 for p in fp16 int4; do (set -x; "$XV/bin/python" -m onnxruntime_genai.models.builder -m Qwen/Qwen2.5-1.5B -i "$MODEL" -o "$OUT/ort/$p" -p "$p" -e cuda -c "$S/ort-cache") > "$S/build-$p.log" 2>&1; done
 rm -rf "$S/ort-cache"; (cd "$OUT" && find ort -type f | sort | xargs sha256sum > ort.sha256)
fi
(cd "$OUT" && sha256sum -c ort.sha256) > "$S/model-check.log"
gox() { local label=$1 v=$2; shift 2; local m=$OUT/ort/fp16; [[ $v == OQ ]] && m=$OUT/ort/int4; (set -x; prun timeout 6000 "$XV/bin/python" "$S/obench.py" "$m" "$S/ref" "$S/out" "$v" "$@") > "$S/$label.log" 2>&1; }
EXT=(OF OQ)
EOF
;;
long) echo 'EXT=(); ATT=long.py';;
*) exit 2;;
esac
cat <<EOF
contexts=($contexts)
EOF
case $stage in
smoke) cat <<'EOF'
c=${contexts[0]}
go validate validate "$S/ref" "$c"
mkdir "$S/out"; go time-GQA time "$S/out" GQA "$c" heldout 1 "$S/ref"
for v in "${EXT[@]}"; do gox "time-$v" "$v" "$c" heldout 1; done
EOF
;;
full) cat <<EOF
ls "\$OUT"/$prior/INSPECTED
EOF
cat <<'EOF'
go validate validate "$S/ref" "${contexts[@]}"
mkdir "$S/out"
ours=(GR GQA); [[ ${#EXT[@]} == 0 ]] && ours=(GR GQ GRA GQA)
for series in 1 2; do
 cs=("${contexts[@]}"); ps=(primary heldout); vs=("${ours[@]}" "${EXT[@]}")
 if [[ $series == 2 ]]; then cs=($(printf '%s\n' "${contexts[@]}" | tac)); ps=(heldout primary); vs=($(printf '%s\n' "${vs[@]}" | tac)); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do
  label="${p}_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  if [[ " ${EXT[*]} " == *" $v "* ]]; then gox "$label" "$v" "$c" "$p" "$series"; else go "$label" time "$S/out" "$v" "$c" "$p" "$series" "$S/ref"; fi
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done; done
done
python3 "$S/analyze.py" "$S/out" "${EXT[@]}" "${contexts[@]}" > "$S/analysis.json"
EOF
;;
*) exit 2;;
esac
