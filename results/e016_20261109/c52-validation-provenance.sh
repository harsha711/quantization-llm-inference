set -euo pipefail
export PROJECT_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109
export SLURM_JOB_ID=332017 SLURM_JOB_NODELIST=c52 SLURM_NODELIST=c52
source "$PROJECT_ROOT/scripts/e016/env_arc.sh"
allocation validation-inspection
[[ -e "$R/VALIDATION_CAPTURED" ]]
sha256sum -c "$R/updated-prerequisite-tooling.sha256" > "$R/validation-prerequisite-source-check.log"
sha256sum build/w4a16_bench build/test_cpu > "$R/build-binaries.sha256"
find scripts/e014 scripts/e015 scripts/e016 src include tests -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$R/validation-source-snapshot.sha256"
tar --exclude='__pycache__' -czf "$R/executed-source-build.tar.gz" scripts/e014 scripts/e015/runner.py scripts/e015/run.py scripts/e015/checks.py scripts/e016 src include tests CMakeLists.txt build/CMakeCache.txt build/CMakeFiles build/w4a16_bench build/test_cpu
find "$R" -type f ! -name 'validation-retrieval.sha256' -print0 | sort -z | xargs -0 sha256sum > /tmp/e016-c52-validation.sha256
cp /tmp/e016-c52-validation.sha256 "$R/validation-retrieval.sha256"
