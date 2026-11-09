set -euo pipefail
export PROJECT_ROOT=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-c9-20261109
export SLURM_JOB_ID=331923 SLURM_JOB_NODELIST=c9 SLURM_NODELIST=c9
source "$PROJECT_ROOT/scripts/e016/env_arc.sh"
allocation inspection
run validation-negative "$E016_PYTHON" scripts/e016/check_analysis.py "$R" validation
run validation-tensors "$E016_PYTHON" scripts/e016/audit_tensors.py "$R" --stage validation
sha256sum build/w4a16_bench build/test_cpu > "$R/build-binaries.sha256"
find scripts/e014 scripts/e015 scripts/e016 src include tests -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$R/validation-source-snapshot.sha256"
tar --exclude='__pycache__' -czf "$R/executed-source-build.tar.gz" scripts/e014 scripts/e015/runner.py scripts/e015/run.py scripts/e015/checks.py scripts/e016 src include tests CMakeLists.txt build/CMakeCache.txt build/CMakeFiles build/w4a16_bench build/test_cpu
cat "$R/validation-negative.log" "$R/validation-tensors.log"
