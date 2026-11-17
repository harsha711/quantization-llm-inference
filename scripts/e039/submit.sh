#!/usr/bin/env bash
# Local helper: emits the E039 batch script (tooling embedded).  scripts/e039/submit.sh prereq|full | ssh arc-off-network sbatch --parsable
set -euo pipefail
mode=${1:?prereq or full}; [[ $mode == prereq || $mode == full ]]
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e039-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=$([[ $mode == prereq ]] && echo 01:30:00 || echo 06:00:00)
#SBATCH --output=/tmp/e039-%j.log
MODE=$mode
EOF
cat <<'EOF'
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e039-20261116
E037=/mnt/beegfs/hpuvvad/csc548-w4a16-e037-20261111
VENV=/tmp/e017-hpuvvad-20261109
LCV=/tmp/e039-lc-hpuvvad-20261116-r1  # attempt 1's overlay (llmcompressor 0.6.0, numpy 1.26.4) is never reused
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/grid.py"; embed "$here/lc.py"
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; nvidia-smi -q; } > "$S/metadata.txt" 2>&1
sha256sum -c "$R/binaries.sha256" > "$S/binary-check.log"
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"; diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
# E037's texts, verified against E037's recorded hashes
for split in test valid; do (cd "$E037" && sha256sum -c wikitext2-$split.txt.sha256); done
grep -q d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0 "$E037/wikitext2-test.txt.sha256"
C4=$E037/c4-train.00000-of-01024.json.gz
echo "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9  $C4" | sha256sum -c
MODEL=$("$E017_PYTHON" -c "import json;print(json.load(open('$R/model.json'))['path'])"); echo "model $MODEL"
# llm-compressor overlay: a fresh venv that sees the frozen E017 packages through a .pth; torch and numpy pinned, its own transformers 5.x (amendment 1)
if [[ ! -e "$LCV/OK" ]]; then
 rm -rf "$LCV"; "$E017_PYTHON" -m venv --without-pip "$LCV"
 SP=$("$E017_PYTHON" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])'); LSP=$("$LCV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
 echo "$SP" > "$LSP/e017-frozen.pth"
 printf 'torch==2.14.0\nnumpy==2.4.6\n' > "$S/constraints.txt"
 PIP_CACHE_DIR=/tmp/e039-pip-cache "$LCV/bin/python" -m pip install -c "$S/constraints.txt" llmcompressor==0.15.0 > "$S/lc-install.log" 2>&1 || { echo INSTALL-FAILED; exit 3; }
 touch "$LCV/OK"
fi
"$LCV/bin/python" -m pip freeze --path "$("$LCV/bin/python" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')" > "$S/lc-overlay-freeze.txt" 2>&1 || true
"$LCV/bin/python" -c 'import torch,transformers,llmcompressor,compressed_tensors as c;print(torch.__version__,torch.__file__,transformers.__version__,llmcompressor.__version__,c.__version__)' > "$S/lc-versions.txt"
grep -q '^2.14.0 ' "$S/lc-versions.txt"; grep -q ' 0.15.0 ' "$S/lc-versions.txt"
"$E017_PYTHON" -m pip freeze | diff -u "$R/pip-freeze.txt" -  # frozen environment untouched by the overlay
H=b3481cea6dd697b8b741ec35d375975d4c48c85d456c884fd314fe6860a7128d
if [[ $MODE == prereq ]]; then NCAL=8; WIN=2; else NCAL=128; WIN=; fi
(set -x; prun timeout 7200 "$LCV/bin/python" "$S/lc.py" "$S/lc.npz" "$MODEL" "$C4" $NCAL) > "$S/lc.log" 2>&1
(set -x; E039_VARIANTS="AT15 AT16" E039_LC="$S/lc.npz" prun timeout 18000 "$E017_PYTHON" "$S/grid.py" "$S" "$E037/wikitext2-test.txt" "$E037/wikitext2-valid.txt" "$C4" $H $NCAL $WIN) > "$S/grid.log" 2>&1
(cd "$S" && sha256sum $(ls | grep -v -E '^(FINISHED|RETRIEVED|driver\.(log|exit))$') > all.sha256)
EOF
