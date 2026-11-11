#!/usr/bin/env bash
# Local helper: emits the E037 batch script (tooling embedded).  scripts/e037/submit.sh | ssh arc-off-network sbatch --parsable
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e037-gptq
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/e037-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e037-20261111
VENV=/tmp/e017-hpuvvad-20261109
mkdir -p "$OUT"; S="$OUT/calib-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/calib.py"
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
# WikiText-2 splits as distributed with the PyTorch examples; the first download fixes the hashes for later runs.
for split in test valid train; do
 if [[ ! -e "$OUT/wikitext2-$split.txt.sha256" ]]; then
  curl -fsSL -o "$OUT/wikitext2-$split.txt" https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$split.txt
  (cd "$OUT" && sha256sum wikitext2-$split.txt > wikitext2-$split.txt.sha256)
 fi
 (cd "$OUT" && sha256sum -c wikitext2-$split.txt.sha256); cp "$OUT/wikitext2-$split.txt.sha256" "$S/"; wc "$OUT/wikitext2-$split.txt" >> "$S/text-wc.txt"
done
grep -q d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0 "$OUT/wikitext2-test.txt.sha256"  # E032's test text
cmp "$OUT/wikitext2-train.txt" /mnt/beegfs/hpuvvad/csc548-w4a16-e036-20261111/wikitext2-train.txt  # E036's calibration text
# Out-of-domain calibration text: the first shard of C4 (allenai/c4, en). Hash fixed before the first download (the hub's LFS object id).
C4=c4-train.00000-of-01024.json.gz
[[ -e "$OUT/$C4" ]] || curl -fsSL -o "$OUT/$C4" https://huggingface.co/datasets/allenai/c4/resolve/main/en/$C4
echo "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9  $OUT/$C4" | sha256sum -c
H=b3481cea6dd697b8b741ec35d375975d4c48c85d456c884fd314fe6860a7128d
ARGS="$OUT/wikitext2-test.txt $OUT/wikitext2-valid.txt $OUT/wikitext2-train.txt $OUT/$C4 $H"
# Cheap prerequisite: 8 calibration windows, 2 windows per split, three variants that cover every code path, 20 sign-SGD steps.
mkdir "$S/smoke"; (set -x; E037_VARIANTS="W/G C/AMT C/S" E037_STEPS=20 prun timeout 3600 "$E017_PYTHON" "$S/calib.py" "$S/smoke" $ARGS 8 2) > "$S/smoke.log" 2>&1
rm "$S/smoke/checkpoint.npz"
[[ -n "${E037_SMOKE_ONLY:-}" ]] && exit 0
(set -x; prun timeout 21600 "$E017_PYTHON" "$S/calib.py" "$S" $ARGS 128) > "$S/calib.log" 2>&1
EOF
