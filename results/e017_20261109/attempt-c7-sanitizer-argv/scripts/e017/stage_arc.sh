#!/usr/bin/env bash
#SBATCH --job-name=w4a16-e017-model
#SBATCH --partition=rtx5060ti16g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:20:00
set -euo pipefail
source "$PROJECT_ROOT/scripts/e017/env_arc.sh"
exec > "$R/model-driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$R/model-driver.exit";touch "$R/MODEL_FINISHED";for ((i=0;i<300;i++));do [[ -e "$R/MODEL_RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
allocation model
[[ -e "$R/ONE_LAYER_INSPECTED" ]]
sha256sum -c "$R/binaries.sha256" > "$R/model-binary-check.log"
"$E017_PYTHON" -m pip freeze > "$R/model-pip-freeze.txt"
diff -u "$R/pip-freeze.txt" "$R/model-pip-freeze.txt" > "$R/model-dependency-diff.txt"
find scripts/e017 scripts/e014 scripts/e004 scripts/e005 src include -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$R/source-validation.sha256"
run native-contract-build c++ -std=c++17 -O2 -fPIC -shared -Iinclude scripts/e017/quant_contract.cpp src/quant.cpp -o /tmp/e017-quant-contract.so
run native-contract-check "$E017_PYTHON" scripts/e017/check_contract.py /tmp/e017-quant-contract.so
run artifact-preparation "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode prepare
names=(model.layers.0.self_attn.q_proj model.layers.0.self_attn.k_proj model.layers.0.mlp.gate_proj model.layers.0.mlp.down_proj)
for name in "${names[@]}";do
 for m in 1 128 2048;do
  run "operator_${name}_M$m" "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode check --name "$name" --m "$m" --all-layers
 done
done
for tool in memcheck synccheck racecheck;do
 filters=();if [[ "$tool" == racecheck ]];then filters=(--kernel-name kns=e005_marlin);fi
 ms=(1 128 2048);if [[ "$tool" == racecheck ]];then ms=(1 128);fi
 for name in "${names[@]}";do
  for m in "${ms[@]}";do
   run "operator_${name}_M$m-$tool" compute-sanitizer --tool "$tool" --error-exitcode 99 "${filters[@]}" "$E017_PYTHON" scripts/e017/prerequisite.py "$R" --mode check --name "$name" --m "$m" --all-layers --suffix "-$tool" --no-trace
  done
 done
done
for context in 128 2048;do
 for variant in O R Q;do
  run "validation_${variant}_${context}" "$E017_PYTHON" scripts/e017/infer.py "$R" --variant "$variant" --context "$context" --stage validation
 done
done
for context in 128 2048;do
 for variant in R Q;do
  run "trace_${variant}_${context}" "$E017_PYTHON" scripts/e017/infer.py "$R" --variant "$variant" --context "$context" --stage trace
 done
 run "trace-coverage-$context" "$E017_PYTHON" scripts/e017/trace_coverage.py "$R" --context "$context"
done
run validation-analysis "$E017_PYTHON" scripts/e017/analyze.py "$R" --stage validation
sha256sum -c "$R/source-validation.sha256" > "$R/validation-source-check.log"
touch "$R/FULL_VALIDATION_CAPTURED"
# Inspection must include actual tensors, dispatch/sanitizers, trace and provenance.
for ((i=0;i<600;i++));do [[ -e "$R/FULL_VALIDATION_INSPECTED" ]] && break;sleep 2;done
[[ -e "$R/FULL_VALIDATION_INSPECTED" ]]
for stage in primary heldout;do
 if [[ "$stage" == heldout ]];then
  [[ -e "$R/PRIMARY_INSPECTED" ]]
  if ! grep -q '"heldout_required": true' "$R/primary-analysis.json";then break;fi
  [[ -e "$R/HELDOUT_RELEASED" ]]
 fi
 sha256sum -c "$R/source-validation.sha256" > "$R/$stage-source-check.log"
 cp "$R/source-validation.sha256" "$R/source-$stage.sha256"
 sha256sum -c "$R/binaries.sha256" > "$R/$stage-binary-check.log"
 "$E017_PYTHON" -m pip freeze > "$R/$stage-pip-freeze.txt"
 diff -u "$R/pip-freeze.txt" "$R/$stage-pip-freeze.txt" > "$R/$stage-dependency-diff.txt"
 for series in 1 2;do
  contexts=(128 2048);variants=(R Q)
  if [[ "$series" == 2 ]];then contexts=(2048 128);variants=(Q R);fi
  for context in "${contexts[@]}";do
   for variant in "${variants[@]}";do
    label="${stage}_${variant}_${context}_s${series}"
    clock > "$R/$label-clocks-before.csv"
    started=$(date +%s%N)
    run "$label" "$E017_PYTHON" scripts/e017/infer.py "$R" --variant "$variant" --context "$context" --stage "$stage" --series "$series"
    finished=$(date +%s%N)
    printf '{"label":"%s","started_utc_ns":%s,"finished_utc_ns":%s}\n' "$label" "$started" "$finished" >> "$R/order-$stage.jsonl"
    clock > "$R/$label-clocks-after.csv"
   done
  done
 done
 run "$stage-analysis" "$E017_PYTHON" scripts/e017/analyze.py "$R" --stage "$stage"
 touch "$R/${stage^^}_CAPTURED"
 for ((i=0;i<600;i++));do [[ -e "$R/${stage^^}_INSPECTED" ]] && break;sleep 2;done
 [[ -e "$R/${stage^^}_INSPECTED" ]]
done
clock > "$R/final-clocks.csv"
sha256sum -c "$R/source-validation.sha256" > "$R/final-source-check.log"
sha256sum -c "$R/binaries.sha256" > "$R/final-binary-check.log"
