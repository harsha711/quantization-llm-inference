#!/usr/bin/env bash
# Local helper: emits an E023 batch script with tooling embedded (ARC home at quota, BeeGFS compute-only).
#   scripts/e023/submit.sh smoke | ssh arc-off-network sbatch --parsable --nodelist=c7
#   scripts/e023/submit.sh validate SMOKEJOB | ... ; scripts/e023/submit.sh time VALIDATEJOB | ...
set -euo pipefail
stage=$1 prior=${2:-none}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e023-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/e023-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e023-20261110
REF=/mnt/beegfs/hpuvvad/csc548-w4a16-e019-20261109/results-333424
mkdir -p "\$OUT"; S="\$OUT/$stage-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
# Hold the allocation after finishing so BeeGFS results can be retrieved over ssh.
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
EOF
embed "$here/../e018/diagnose.py"; embed "$here/../e021/validate.py"; embed "$here/run.py"; embed "$here/analyze.py"
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
go() { local label=$1; shift; (set -x; prun timeout 3000 "$E017_PYTHON" "$S/run.py" "$@") > "$S/$label.log" 2>&1; }
EOF
case $stage in
smoke) echo 'go validate validate "$S/out" 8 128' ;;
validate) cat <<EOF
ls "\$OUT/smoke-$prior/INSPECTED"
go validate validate "\$S/out" 64 128 2048
EOF
;;
time) cat <<EOF
ls "\$OUT/validate-$prior/INSPECTED"
A="\$OUT/validate-$prior/out"
EOF
cat <<'EOF'
mkdir "$S/out"
for series in 1 2; do
 contexts=(128 2048); variants=(R GR Q GQ)
 if [[ $series == 2 ]]; then contexts=(2048 128); variants=(GQ Q GR R); fi
 for c in "${contexts[@]}"; do for v in "${variants[@]}"; do
  label="primary_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  go "$label" time "$REF" "$S/out" "$v" "$c" "$series" "$A"
  clock > "$S/$label-clocks-after.csv"
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done
done
python3 "$S/analyze.py" "$S/out" > "$S/analysis.json"
EOF
;;
*) exit 2;;
esac
