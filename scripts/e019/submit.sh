#!/usr/bin/env bash
# Local helper: emits an E019 batch script with tooling embedded (ARC home is at quota, BeeGFS is compute-only).
#   scripts/e019/submit.sh smoke | ssh arc-off-network sbatch --parsable --nodelist=c7
#   scripts/e019/submit.sh validate SMOKEJOB | ... ; scripts/e019/submit.sh primary|heldout VALIDATEJOB | ...
set -euo pipefail
stage=$1 prior=${2:-none}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
embed() { printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$(basename "$1")"; cat "$1"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$1" | cut -d' ' -f1)" "$(basename "$1")"; }
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e019-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/tmp/e019-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e019-20261109
mkdir -p "\$OUT"; S="\$OUT/$stage-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
# Hold the allocation after finishing so BeeGFS results can be retrieved over ssh.
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
EOF
embed "$here/../e018/diagnose.py"; embed "$here/validate.py"; embed "$here/analyze.py"
cat <<'EOF'
export PROJECT_ROOT=$E017_ROOT
set +u; source "$E017_ROOT/scripts/e017/env_arc.sh"; set -u
export TRITON_CACHE_DIR="$OUT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor" XDG_CACHE_HOME="$OUT/cache/xdg" CUDA_CACHE_PATH="$OUT/cache/cuda"
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
go() { local label=$1; shift; (set -x; prun timeout 3000 "$@") > "$S/$label.log" 2>&1; }
infer() { go "$1" "$E017_PYTHON" "$E017_ROOT/scripts/e017/infer.py" "$V" "${@:2}"; }
EOF
case $stage in
smoke) cat <<'EOF'
for c in 128 2048; do go "validate-$c" "$E017_PYTHON" "$S/validate.py" "$S/out-$c" "$c" 8; done
EOF
;;
validate) cat <<EOF
ls "\$OUT/smoke-$prior/INSPECTED"
V="\$OUT/results-\$SLURM_JOB_ID"; mkdir "\$V"; ln -s "\$R/model.json" "\$R/artifacts" "\$V/"
rc=0
for c in 128 2048; do
 for v in O R; do infer "infer-validation-\$v-\$c" --variant "\$v" --context "\$c" --stage validation; done
 go "validate-\$c" "\$E017_PYTHON" "\$S/validate.py" "\$S/out-\$c" "\$c" 64 "\$V" || rc=1
done
exit \$rc
EOF
;;
primary|heldout) cat <<EOF
ls "\$OUT/validate-$prior/INSPECTED"
V="\$OUT/results-$prior"; stage=$stage
EOF
cat <<'EOF'
touch "$V/FULL_VALIDATION_INSPECTED"
if [[ $stage == heldout ]]; then grep -q '"heldout_required": true' "$OUT"/primary-*/primary-analysis.json; touch "$V/HELDOUT_RELEASED"; fi
for series in 1 2; do
 contexts=(128 2048); variants=(R Q)
 if [[ $series == 2 ]]; then contexts=(2048 128); variants=(Q R); fi
 for c in "${contexts[@]}"; do for v in "${variants[@]}"; do
  label="${stage}_${v}_${c}_s${series}"
  clock > "$S/$label-clocks-before.csv"
  infer "$label" --variant "$v" --context "$c" --stage "$stage" --series "$series"
  clock > "$S/$label-clocks-after.csv"
  printf '{"label":"%s","finished_utc_ns":%s}\n' "$label" "$(date +%s%N)" >> "$S/order.jsonl"
 done; done
done
python3 "$S/analyze.py" "$V" "$stage" > "$S/$stage-analysis.json"
cp "$V/$stage"_*.jsonl "$V/$stage"_*.json "$S/"
EOF
;;
*) exit 2;;
esac
