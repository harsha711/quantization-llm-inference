#!/usr/bin/env bash
# Local helper: emits the E018 batch script (diagnose.py embedded, because ARC home is at quota
# and BeeGFS is compute-node-only). Usage:
#   scripts/e018/submit.sh smoke|full | ssh arc-off-network sbatch --parsable --nodelist=c7 [--dependency=afterok:ID]
set -euo pipefail
stage=$1; [[ $stage == smoke || $stage == full ]]
gen=64; [[ $stage == smoke ]] && gen=8
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e018-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=/tmp/e018-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e018-20261109
mkdir -p "\$OUT"; S="\$OUT/$stage-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
# Hold the allocation after finishing so BeeGFS results can be retrieved over ssh.
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<300;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
[[ $stage == smoke ]] || ls "\$OUT"/smoke-*/INSPECTED
cat > "\$S/diagnose.py" <<'PYEOF'
$(cat "$here/diagnose.py")
PYEOF
echo "$(sha256sum < "$here/diagnose.py" | cut -d' ' -f1)  \$S/diagnose.py" | sha256sum -c
export PROJECT_ROOT=\$E017_ROOT
set +u; source "\$E017_ROOT/scripts/e017/env_arc.sh"; set -u
scontrol show job "\$SLURM_JOB_ID" > "\$S/allocation.txt"
grep -q 'UserId=hpuvvad' "\$S/allocation.txt"; grep -q 'JobState=RUNNING' "\$S/allocation.txt"
[[ \$(hostname) == "\$SLURM_JOB_NODELIST" ]]
prun /bin/hostname > "\$S/launcher.log" 2>&1
[[ \$(grep -c "^\$SLURM_JOB_NODELIST\$" "\$S/launcher.log") == 1 ]]
{ date --iso-8601=seconds; hostname; lscpu; nvidia-smi -q; env | grep -E '^(SLURM_|OMP_|OMPI_)' | sort; } > "\$S/metadata.txt" 2>&1
sha256sum -c "\$R/binaries.sha256" > "\$S/binary-check.log"
"\$E017_PYTHON" -m pip freeze > "\$S/pip-freeze.txt"
diff -u "\$R/pip-freeze.txt" "\$S/pip-freeze.txt"
(cd "\$E017_ROOT" && sha256sum -c "\$R/source-validation.sha256") > "\$S/source-check.log"
(set -x; prun timeout 1800 "\$E017_PYTHON" "\$S/diagnose.py" "\$S/out" $gen) > "\$S/diagnose.log" 2>&1
EOF
