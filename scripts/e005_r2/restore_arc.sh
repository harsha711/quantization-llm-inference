#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e005_r2/env_arc.sh"
[[ -e "$R/e006-preserved-binaries-check.log" ]]
[[ ! -e /tmp/e005-r2-hpuvvad-20261106 ]]
run venv python3.11 -m venv /tmp/e005-r2-hpuvvad-20261106
# Restore the historical dependency set, with dependency resolution disabled.
run pip "$E005_PYTHON" -m pip install --no-deps -r "$E004_CONTROL/results/e004_20261106/pip-freeze.txt"
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E004_CONTROL/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
bench_base
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run identity "$E005_PYTHON" scripts/e005_r2/identity.py
run smoke "${base[@]}" --out "$R/smoke.jsonl" --shapes 256,2048 --patterns mixed zero_a zero_w \
 --warmup 0 --iters 0 --policies no_flush
sha256sum "$E005_LOCAL" "$E005_BRIDGE" "$E005_WHEEL" scripts/e005/bench.py scripts/e004/bench.py > "$R/verified-inputs.sha256"
touch "$R/SMOKE_PASSED"
