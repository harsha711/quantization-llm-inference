#!/usr/bin/env bash
# Local helper: emits an E027 batch script (tooling embedded). The job runs on whichever node Slurm gives it.
#   scripts/e027/submit.sh smoke | ssh arc-off-network sbatch --parsable
#   scripts/e027/submit.sh full SMOKEDIR "128 512 2048" | ssh arc-off-network sbatch --parsable --nodelist=<smoke node>
set -euo pipefail
stage=$1 prior=${2:-none} contexts=${3:-128} att=${4:-att.py}  # att: long.py for contexts above 2048
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
mkdir -p "\$OUT"; S="\$OUT/$stage-\$(hostname)-\$SLURM_JOB_ID\${E_TAG:-}"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
EOF
cat <<'EOF'
# Same runtime bytes as c7, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/../e021/validate.py"; embed "$here/../e023/run.py"; embed "$here/../e026/att.py"
embed "$here/../e028/long.py"; embed "$here/../e030/gprofile.py"; embed "$here/../e030/vprofile.py"; embed "$here/../e031/head.py"; embed "$here/../e031/headconv.py"; embed "$here/../e031/headanalyze.py"; embed "$here/../e028/obench.py"; printf 'cat > "$S/e033analyze.py" <<'"'"'PYEOF'"'"'\n'; cat "$here/../e033/analyze.py"; printf 'PYEOF\n'; embed "$here/convert.py"; embed "$here/vbench.py"; embed "$here/analyze.py"
embed "$here/../e038/ckpt.py"; printf 'cat > "$S/e038analyze.py" <<'"'"'PYEOF'"'"'\n'; cat "$here/../e038/analyze.py"; printf 'PYEOF\n'
embed "$here/../e035/head8conv.py"; printf 'cat > "$S/e035analyze.py" <<'"'"'PYEOF'"'"'\n'; cat "$here/../e035/analyze.py"; printf 'PYEOF\n'
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
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CONFIG_ROOT="$S/vllm-config" VLLM_CACHE_ROOT="$S/vllm-cache"
WHEEL=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-20261109/external/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl
echo "4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa  $WHEEL" | sha256sum -c
if [[ ! -e "$VV/INSTALLED" ]]; then
 rm -rf "$VV"; python3.11 -m venv "$VV"
 if [[ -e "$OUT/vllm-freeze.txt" ]]; then "$VV/bin/python" -m pip install "$WHEEL" -c "$OUT/vllm-freeze.txt" > "$S/vllm-install.log" 2>&1
 else "$VV/bin/python" -m pip install "$WHEEL" > "$S/vllm-install.log" 2>&1; fi
 touch "$VV/INSTALLED"
fi
"$VV/bin/python" -m pip freeze > "$S/vllm-freeze.txt"
[[ -e "$OUT/vllm-freeze.txt" ]] || cp "$S/vllm-freeze.txt" "$OUT/vllm-freeze.txt"
diff -u "$OUT/vllm-freeze.txt" "$S/vllm-freeze.txt"
go() { local label=$1; shift; (set -x; prun timeout 6000 "$E017_PYTHON" "$S/$ATT" "$@") > "$S/$label.log" 2>&1; }
gov() { local label=$1; shift; (set -x; prun timeout 3000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq" "$S/ref" "$S/out" "$@") > "$S/$label.log" 2>&1; }
if [[ ! -e "$OUT/model-gptq.sha256" ]]; then
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/convert.py" "$OUT/model-gptq") > "$S/convert.log" 2>&1
 (cd "$OUT" && sha256sum model-gptq/* > model-gptq.sha256)
fi
(cd "$OUT" && sha256sum -c model-gptq.sha256) > "$S/model-check.log"
EOF
case $stage in
smoke) cat <<'EOF'
ATT=att.py
go validate validate "$S/ref" 128
mkdir "$S/out"; go time-GQA time "$S/out" GQA 128 heldout 1 "$S/ref"
gov time-V 128 heldout 1
EOF
;;
full) cat <<EOF
ls "\$OUT"/$prior/INSPECTED
contexts=($contexts)
ATT=$att
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
profile) cat <<'EOF'
# E030 diagnostic: per-kernel device time per decode step, ours then vLLM, same node and weights
(set -x; prun timeout 3000 "$E017_PYTHON" "$S/gprofile.py" "$S/prof" 128 2048) > "$S/profile-ours.log" 2>&1
(set -x; prun timeout 3000 "$VV/bin/python" "$S/vprofile.py" "$OUT/model-gptq" "$S/prof" 128 2048) > "$S/profile-vllm.log" 2>&1
EOF
;;
head-smoke|head-full) cat <<'EOF'
# E031: 4-bit output head. Checkpoint for vLLM = the E027 checkpoint plus the quantized head.
if [[ ! -e "$OUT/model-gptq-head.sha256" ]]; then
 rm -rf "$OUT/model-gptq-head"
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/headconv.py" "$OUT/model-gptq" "$OUT/model-gptq-head") > "$S/headconv.log" 2>&1
 (cd "$OUT" && sha256sum model-gptq-head/* > model-gptq-head.sha256)
fi
(cd "$OUT" && sha256sum -c model-gptq-head.sha256) > "$S/head-model-check.log"
goh() { local label=$1; shift; (set -x; prun timeout 6000 "$E017_PYTHON" "$S/head.py" "$@") > "$S/$label.log" 2>&1; }
govh() { local label=$1 dir=$2; shift 2; (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/$dir" "$S/ref" "$S/out" "$@") > "$S/$label.log" 2>&1; }
cell() { local label=$1 v=$2 c=$3 p=$4 series=$5
 case $v in
 V) govh "$label" model-gptq "$c" "$p" "$series" V GQA;;
 VH) govh "$label" model-gptq-head "$c" "$p" "$series" VH GQAH;;
 *) goh "$label" time "$S/out" "$v" "$c" "$p" "$series" "$S/ref";;
 esac; }
EOF
if [[ $stage == head-smoke ]]; then cat <<'EOF'
goh validate validate "$S/ref" 128
mkdir "$S/out"
for v in GQAH V VH; do cell "time-$v" "$v" 128 heldout 1; done
EOF
else cat <<EOF
ls "\$OUT"/$prior/INSPECTED
contexts=($contexts)
EOF
cat <<'EOF'
goh validate validate "$S/ref" "${contexts[@]}"
mkdir "$S/out"
for series in 1 2; do
 cs=("${contexts[@]}"); ps=(primary heldout); vs=(GQA GQAH V VH)
 if [[ $series == 2 ]]; then cs=($(printf '%s\n' "${contexts[@]}" | tac)); ps=(heldout primary); vs=(VH V GQAH GQA); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do
  label="${p}_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  cell "$label" "$v" "$c" "$p" "$series"
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done; done
done
python3 "$S/headanalyze.py" "$S/out" "${contexts[@]}" > "$S/analysis.json"
EOF
fi
;;
e033) cat <<'EOF'
# E033: vLLM+head versus ORT INT4 interleaved in one job, and the head gain in vLLM by batch size. Needs both node-local venvs.
XV=/tmp/e028-ort-hpuvvad-20261110; [[ -e "$XV/INSTALLED" ]]
(cd "$OUT" && sha256sum -c model-gptq-head.sha256) > "$S/head-model-check.log"
(cd /mnt/beegfs/hpuvvad/csc548-w4a16-e028-20261110 && sha256sum -c ort.sha256) > "$S/ort-model-check.log"
"$XV/bin/python" -m pip freeze > "$S/ort-freeze.txt"; diff -u /mnt/beegfs/hpuvvad/csc548-w4a16-e028-20261110/ort-freeze.txt "$S/ort-freeze.txt"
(set -x; prun timeout 6000 "$E017_PYTHON" "$S/head.py" validate "$S/ref" 128 2048) > "$S/validate.log" 2>&1
mkdir "$S/out"
cell() { local p=$1 v=$2 c=$3 series=$4 b=$5 label
 label="${p}_${v}_${c}_s${series}"
 clock > "$S/$label-clocks-before.csv"
 case $v in
 OQ) (set -x; prun timeout 6000 "$XV/bin/python" "$S/obench.py" /mnt/beegfs/hpuvvad/csc548-w4a16-e028-20261110/ort/int4 "$S/ref" "$S/out" OQ "$c" "$p" "$series") > "$S/$label.log" 2>&1;;
 VH*) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq-head" "$S/ref" "$S/out" "$c" "$p" "$series" "$v" GQAH "$b") > "$S/$label.log" 2>&1;;
 V*) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq" "$S/ref" "$S/out" "$c" "$p" "$series" "$v" GQA "$b") > "$S/$label.log" 2>&1;;
 esac
 printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"; }
for series in 1 2; do
 cs=(128 2048); ps=(primary heldout); vs=(V VH OQ); bs=(2 4 8)
 if [[ $series == 2 ]]; then cs=(2048 128); ps=(heldout primary); vs=(OQ VH V); bs=(8 4 2); fi
 for c in "${cs[@]}"; do
  for p in "${ps[@]}"; do for v in "${vs[@]}"; do cell "$p" "$v" "$c" "$series" 1; done; done
  for b in "${bs[@]}"; do for v in "${vs[@]}"; do [[ $v == OQ ]] || cell primary "${v}b$b" "$c" "$series" "$b"; done; done
 done
done
python3 "$S/e033analyze.py" "$S/out" > "$S/analysis.json"
EOF
;;
e035) cat <<'EOF'
# E035: 8-bit output head in vLLM against the FP16 head (V) and the 4-bit head (VH), interleaved in one job.
(cd "$OUT" && sha256sum -c model-gptq-head.sha256) > "$S/head-model-check.log"
if [[ ! -e "$OUT/model-gptq-head8.sha256" ]]; then
 rm -rf "$OUT/model-gptq-head8"
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/head8conv.py" "$OUT/model-gptq" "$OUT/model-gptq-head8") > "$S/head8conv.log" 2>&1
 (cd "$OUT" && sha256sum model-gptq-head8/* > model-gptq-head8.sha256)
fi
(cd "$OUT" && sha256sum -c model-gptq-head8.sha256) > "$S/head8-model-check.log"
(set -x; prun timeout 6000 "$E017_PYTHON" "$S/head.py" validate "$S/ref" 128 2048) > "$S/validate.log" 2>&1
cell() { local dir=$1 p=$2 v=$3 c=$4 series=$5 m=model-gptq key=GQA label
 label="${p}_${v}_${c}_s${series}"
 case $v in VH) m=model-gptq-head; key=GQAH;; VH8) m=model-gptq-head8;; esac  # VH8 is compared with the FP16-head tokens
 clock > "$S/$dir-$label-clocks-before.csv"
 (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/$m" "$S/ref" "$S/$dir" "$c" "$p" "$series" "$v" "$key") > "$S/$dir-$label.log" 2>&1
 printf '{"label":"%s/%s","finished_utc_ns":%s}\n' "$dir" "$label" "$(date +%s%N)" >> "$S/order.jsonl"; }
# Cheap prerequisite: one 8-bit-head cell must load, match the first reference token and agree on at least 32 leading tokens.
mkdir "$S/smoke" "$S/out"; cell smoke heldout VH8 128 1
python3 -c "import json,sys; a=json.load(open(sys.argv[1]))['tokens_agree_min']; print('smoke agree', a); assert a >= 32" "$S/smoke/heldout_VH8_128_s1.json"
grep -h "AutoGPTQLinearMethod\|lm_head\|uint8b128" "$S/smoke-heldout_VH8_128_s1.log" | head -20 > "$S/smoke-kernel-lines.txt" || true
for series in 1 2; do
 cs=(128 2048); ps=(primary heldout); vs=(V VH VH8)
 if [[ $series == 2 ]]; then cs=(2048 128); ps=(heldout primary); vs=(VH8 VH V); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do cell out "$p" "$v" "$c" "$series"; done; done; done
done
python3 "$S/e035analyze.py" "$S/out" > "$S/analysis.json"
EOF
;;
e038) cat <<EOF
CK=$prior  # E037 run directory holding checkpoint.npz and quality.json
EOF
cat <<'EOF'
# E038: gate and time E037's calibrated checkpoint against the plain-rounded tensors, in our path and in vLLM, one job.
X=/mnt/beegfs/hpuvvad/csc548-w4a16-e038-20261111; mkdir -p "$X"
export E038_CKPT="$CK/checkpoint.npz"
echo "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint_sha256'])" "$CK/quality.json")  $E038_CKPT" | sha256sum -c
cp "$CK/quality.json" "$S/e037-quality.json"
(cd "$OUT" && sha256sum -c model-gptq-head.sha256) > "$S/head-model-check.log"
if [[ ! -e "$X/built.sha256" ]]; then
 rm -rf "$X/root" "$X/model-gptq-c" "$X/model-gptq-c-head"
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/ckpt.py" root "$X/root") > "$S/root.log" 2>&1
 (set -x; E017_ROOT="$X/root" prun timeout 3000 "$E017_PYTHON" "$S/convert.py" "$X/model-gptq-c") > "$S/convert-c.log" 2>&1
 (set -x; prun timeout 3000 "$E017_PYTHON" "$S/ckpt.py" headconv "$X/model-gptq-c" "$X/model-gptq-c-head") > "$S/headconv-c.log" 2>&1
 (cd "$X" && sha256sum model-gptq-c/* model-gptq-c-head/* root/results/e017_20261109/artifacts/* > built.sha256)
fi
(cd "$X" && sha256sum -c built.sha256) > "$S/built-check.log"
cell() { local dir=$1 p=$2 v=$3 c=$4 series=$5 label
 label="${p}_${v}_${c}_s${series}"
 clock > "$S/$dir-$label-clocks-before.csv"
 case $v in
 GQA|GQAH) (set -x; prun timeout 6000 "$E017_PYTHON" "$S/head.py" time "$S/$dir" "$v" "$c" "$p" "$series" "$S/ref") > "$S/$dir-$label.log" 2>&1;;
 cGQA|cGQAH) (set -x; E017_ROOT="$X/root" prun timeout 6000 "$E017_PYTHON" "$S/ckpt.py" head time "$S/${dir}c" "${v#c}" "$c" "$p" "$series" "$S/refc") > "$S/$dir-$label.log" 2>&1;;
 V) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq" "$S/ref" "$S/$dir" "$c" "$p" "$series" V GQA) > "$S/$dir-$label.log" 2>&1;;
 VH) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$OUT/model-gptq-head" "$S/ref" "$S/$dir" "$c" "$p" "$series" VH GQAH) > "$S/$dir-$label.log" 2>&1;;
 cV) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$X/model-gptq-c" "$S/refc" "$S/${dir}c" "$c" "$p" "$series" V GQA) > "$S/$dir-$label.log" 2>&1;;
 cVH) (set -x; prun timeout 6000 "$VV/bin/python" "$S/vbench.py" "$X/model-gptq-c-head" "$S/refc" "$S/${dir}c" "$c" "$p" "$series" VH GQAH) > "$S/$dir-$label.log" 2>&1;;
 esac
 printf '{"label":"%s/%s","finished_utc_ns":%s}\n' "$dir" "$label" "$(date +%s%N)" >> "$S/order.jsonl"; }
# The gate, plain then calibrated (E031's validate: E021 criteria against an FP64 control on the same weights and head).
(set -x; prun timeout 6000 "$E017_PYTHON" "$S/head.py" validate "$S/ref" 128 2048) > "$S/validate.log" 2>&1
(set -x; E017_ROOT="$X/root" prun timeout 6000 "$E017_PYTHON" "$S/ckpt.py" head validate "$S/refc" 128 2048) > "$S/validate-c.log" 2>&1
# Cheap prerequisite: one calibrated cell per stack must load, match its reference and write a row.
mkdir "$S/smoke" "$S/smokec" "$S/out" "$S/outc"
cell smoke heldout cGQAH 128 1; cell smoke heldout cVH 128 1
ls "$S/smokec/heldout_GQAH_128_s1.json" "$S/smokec/heldout_VH_128_s1.json"
for series in 1 2; do
 cs=(128 2048); ps=(primary heldout); vs=(GQA cGQA GQAH cGQAH V cV VH cVH)
 if [[ $series == 2 ]]; then cs=(2048 128); ps=(heldout primary); vs=(cVH VH cV V cGQAH GQAH cGQA GQA); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do cell out "$p" "$v" "$c" "$series"; done; done; done
done
python3 "$S/e038analyze.py" "$S/out" "$S/outc" > "$S/analysis.json"
EOF
;;
*) exit 2;;
esac
