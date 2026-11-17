#!/usr/bin/env bash
# Local helper: emits the E042 batch script (tooling embedded).  scripts/e042/submit.sh prereq|full | ssh arc-off-network sbatch --parsable
set -euo pipefail
mode=${1:?prereq or full}; [[ $mode == prereq || $mode == full ]]
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e042-$mode
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=$([[ $mode == prereq ]] && echo 00:45:00 || echo 05:00:00)
#SBATCH --output=/tmp/e042-%j.log
MODE=$mode
EOF
cat <<'EOF'
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e042-20261117
E037=/mnt/beegfs/hpuvvad/csc548-w4a16-e037-20261111
VENV=/tmp/e017-hpuvvad-20261109
mkdir -p "$OUT"; S="$OUT/$MODE-$(hostname)-$SLURM_JOB_ID"; mkdir "$S"
exec > "$S/driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$S/driver.exit";touch "$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "$S/RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
embed "$here/../e018/diagnose.py"; embed "$here/sens.py"
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
for split in test valid; do (cd "$E037" && sha256sum -c wikitext2-$split.txt.sha256); done
grep -q d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0 "$E037/wikitext2-test.txt.sha256"
H=b3481cea6dd697b8b741ec35d375975d4c48c85d456c884fd314fe6860a7128d
if [[ $MODE == prereq ]]; then WIN=2; else WIN=; fi
(set -x; prun timeout 16200 "$E017_PYTHON" "$S/sens.py" "$S" "$E037/wikitext2-test.txt" "$E037/wikitext2-valid.txt" $H $WIN) > "$S/sens.log" 2>&1
(cd "$S" && sha256sum $(ls | grep -v -E '^(FINISHED|RETRIEVED|driver\.(log|exit))$') > all.sha256)
EOF
