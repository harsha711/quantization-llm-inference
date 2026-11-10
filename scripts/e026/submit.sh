#!/usr/bin/env bash
# Local helper: emits an E024 batch script (tooling embedded). NODE is the node the job will be pinned to.
#   scripts/e026/submit.sh smoke c7 | ssh arc-off-network sbatch --parsable --nodelist=c7
#   scripts/e026/submit.sh full c7 SMOKEJOB "128 256 512 1024 2048" | ...
#   other nodes unpack the hashed E024 runtime archive to the identical /tmp path first
set -euo pipefail
stage=$1 node=$2 prior=${3:-none} contexts=${4:-128}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e026-$stage-$node
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --output=/tmp/e026-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e026-20261110
VENV=/tmp/e017-hpuvvad-20261109
mkdir -p "\$OUT"; S="\$OUT/$stage-$node-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
[[ \$(hostname) == $node ]]
EOF
if [[ $node != c7 ]]; then cat <<'EOF'
# Same runtime bytes as c7, unpacked to the identical node-local path (the venv is not relocatable).
T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "$VENV/bin/python" ]]; then sha256sum -c "$T.sha256"; tar xf "$T" -C /tmp; fi
EOF
fi
embed "$here/../e018/diagnose.py"; embed "$here/../e021/validate.py"; embed "$here/../e023/run.py"; embed "$here/att.py"; embed "$here/analyze.py"
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
go() { local label=$1; shift; (set -x; prun timeout 3000 "$E017_PYTHON" "$S/att.py" "$@") > "$S/$label.log" 2>&1; }
EOF
case $stage in
smoke) cat <<'EOF'
go validate validate "$S/ref" 128
mkdir "$S/out"; go time-GQA time "$S/out" GQA 128 heldout 1 "$S/ref"
EOF
;;
full) cat <<EOF
ls "\$OUT"/smoke-$node-$prior/INSPECTED
contexts=($contexts)
EOF
cat <<'EOF'
go validate validate "$S/ref" "${contexts[@]}"
mkdir "$S/out"
for series in 1 2; do
 cs=("${contexts[@]}"); ps=(primary heldout); vs=(GR GQ GRA GQA)
 if [[ $series == 2 ]]; then cs=($(printf '%s\n' "${contexts[@]}" | tac)); ps=(heldout primary); vs=(GQA GRA GQ GR); fi
 for c in "${cs[@]}"; do for p in "${ps[@]}"; do for v in "${vs[@]}"; do
  label="${p}_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  go "$label" time "$S/out" "$v" "$c" "$p" "$series" "$S/ref"
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done; done
done
python3 "$S/analyze.py" "$S/out" "${contexts[@]}" > "$S/analysis.json"
EOF
;;
*) exit 2;;
esac
