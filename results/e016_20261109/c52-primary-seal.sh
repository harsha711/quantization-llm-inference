set -euo pipefail
export PROJECT_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109
export SLURM_JOB_ID=332017 SLURM_JOB_NODELIST=c52 SLURM_NODELIST=c52
source "$PROJECT_ROOT/scripts/e016/env_arc.sh"
allocation final-inspection
[[ -e "$R/PRIMARY_CAPTURED" && $(cat "$R/primary-driver.exit") == 0 ]]
grep -q '"heldout_required": false' "$R/primary-analysis.json"
[[ ! -e "$R/HELDOUT_CAPTURED" ]]
sha256sum -c "$R/validation-tooling.sha256" > "$R/final-source-check.log"
clock > "$R/final-clocks.csv"
find "$R" -maxdepth 1 -type f -name 'primary*' -print0 | sort -z | xargs -0 sha256sum > "$R/primary-retrieval.sha256"
touch "$R/PRIMARY_INSPECTED"
