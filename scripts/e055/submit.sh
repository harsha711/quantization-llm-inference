#!/usr/bin/env bash
# Local helper: emits the E055 batch script (tooling embedded).  scripts/e055/submit.sh | ssh arc-off-network sbatch --parsable --nodelist=cN
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e055-quality
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/tmp/e055-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e055-20261119
E32=/mnt/beegfs/hpuvvad/csc548-w4a16-e032-20261110
VENV=/tmp/e017-hpuvvad-20261109
export E052_LIB=/mnt/beegfs/hpuvvad/csc548-w4a16-e050-20261118/prereq-c8-343302/libe050_tc.so
mkdir -p "$OUT"; S="$OUT/quality-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";exit "$rc"' EXIT
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then echo restored > "$S/runtime-restored.txt"; sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
for f in ../e018/diagnose.py quality.py ../e052/t.py analyze.py; do embed "$here/$f"; done
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "$SLURM_JOB_ID" > "$S/allocation.txt"
grep -q 'UserId=hpuvvad' "$S/allocation.txt"; grep -q 'JobState=RUNNING' "$S/allocation.txt"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader) == 'NVIDIA GeForce RTX 5060 Ti' ]]
prun /bin/hostname > "$S/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; nvidia-smi -q; } > "$S/metadata.txt" 2>&1
sha256sum -c "$R/binaries.sha256" > "$S/binary-check.log"
echo "a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe  $E052_LIB" | sha256sum -c > "$S/lib-check.log"
"$E017_PYTHON" -m pip freeze > "$S/pip-freeze.txt"; diff -u "$R/pip-freeze.txt" "$S/pip-freeze.txt"
(cd "$E017_ROOT" && sha256sum -c "$R/source-validation.sha256") > "$S/source-check.log"
TEXT="$E32/wikitext2-test.txt"; (cd "$E32" && sha256sum -c wikitext2-test.txt.sha256) > "$S/text-check.log"; cp "$E32/wikitext2-test.txt.sha256" "$S/"
H=b3481cea6dd697b8b741ec35d375975d4c48c85d456c884fd314fe6860a7128d  # unused for blocks-only variants (E032 argument kept)
q() { local label=$1; shift; (set -x; prun timeout 3000 "$E017_PYTHON" "$@") > "$S/quality-$label.log" 2>&1; }
q R  "$S/quality.py" "$S" R "$TEXT" $H
q QM "$S/t.py" --marlin quality.py "$S" Q "$TEXT" $H QM
q QT "$S/t.py" quality.py "$S" Q "$TEXT" $H QT
q O  "$S/quality.py" "$S" O "$TEXT" $H
"$E017_PYTHON" "$S/analyze.py" "$S" > "$S/analysis.json"
(cd "$S" && find . -type f ! -name FINISHED ! -name 'driver.*' ! -name all.sha256 | sort | xargs sha256sum > all.sha256)
echo STAGE-OK
EOF
