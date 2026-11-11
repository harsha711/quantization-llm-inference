#!/usr/bin/env bash
# Local helper: emits the E032 batch script (tooling embedded).  scripts/e032/submit.sh | ssh arc-off-network sbatch --parsable
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e032-quality
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/e032-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e032-20261110
VENV=/tmp/e017-hpuvvad-20261109
mkdir -p "$OUT"; S="$OUT/quality-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/quality.py"
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
# WikiText-2 test split as distributed with the PyTorch examples; the first download fixes the hash for later runs.
TEXT="$OUT/wikitext2-test.txt"
if [[ ! -e "$TEXT.sha256" ]]; then
 curl -fsSL -o "$TEXT" https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt
 (cd "$OUT" && sha256sum wikitext2-test.txt > wikitext2-test.txt.sha256)
fi
(cd "$OUT" && sha256sum -c wikitext2-test.txt.sha256); cp "$OUT/wikitext2-test.txt.sha256" "$S/"; wc "$TEXT" > "$S/text-wc.txt"
for v in O OH R RH; do (set -x; prun timeout 1500 "$E017_PYTHON" "$S/quality.py" "$S" "$v" "$TEXT" b3481cea6dd697b8b741ec35d375975d4c48c85d456c884fd314fe6860a7128d) > "$S/quality-$v.log" 2>&1; done
EOF
