#!/usr/bin/env bash
# Reuse the verified one-rank environment and log/clock helpers without old gates.
source "$PROJECT_ROOT/scripts/e005_r1/env_arc.sh"
export E007_RESULTS="$PROJECT_ROOT/results/e007_20261106"
R="$E007_RESULTS"
base=("$E005_PYTHON" scripts/e007/bench.py)
