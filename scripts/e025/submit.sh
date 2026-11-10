#!/usr/bin/env bash
# Local helper: emits the E021 batch script (tooling embedded, because ARC home is at quota
# and BeeGFS is compute-node-only). Usage:
#   scripts/e025/submit.sh smoke|full | ssh arc-off-network sbatch --parsable --nodelist=c7 [--dependency=afterok:ID]
set -euo pipefail
stage=$1; [[ $stage == smoke || $stage == full ]]
gen= contexts="128 2048"; [[ $stage == smoke ]] && contexts=128
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e025-$stage
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/e025-$stage-%j.log
set -euo pipefail
export E017_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e017-r2-20261109
OUT=/mnt/beegfs/hpuvvad/csc548-w4a16-e025-20261110
mkdir -p "\$OUT"; S="\$OUT/$stage-\$SLURM_JOB_ID"; mkdir "\$S"
exec > "\$S/driver.log" 2>&1
# Hold the allocation after finishing so BeeGFS results can be retrieved over ssh.
trap 'rc=\$?;echo "\$rc" > "\$S/driver.exit";touch "\$S/FINISHED";for ((i=0;i<450;i++));do [[ -e "\$S/RETRIEVED" ]] && break;sleep 2;done;exit "\$rc"' EXIT
[[ $stage == smoke ]] || ls "\$OUT"/smoke-*/INSPECTED
$(for f in "$here/../e018/diagnose.py" "$here/opprofile.py"; do b=$(basename "$f"); printf 'cat > "$S/%s" <<'"'"'PYEOF'"'"'\n' "$b"; cat "$f"; printf 'PYEOF\necho "%s  $S/%s" | sha256sum -c\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$b"; done)
# Nodes other than c7 get the hashed c7 runtime archive (E024) unpacked to the identical node-local path.
V=/tmp/e017-hpuvvad-20261109; T=/mnt/beegfs/hpuvvad/csc548-w4a16-e024-20261110/runtime.tar
if [[ ! -e "\$V/bin/python" ]]; then sha256sum -c "\$T.sha256"; tar xf "\$T" -C /tmp; fi
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
(set -x; prun timeout 3000 "\$E017_PYTHON" "\$S/opprofile.py" "\$S/out" $gen $contexts) > "\$S/opprofile.log" 2>&1
EOF
