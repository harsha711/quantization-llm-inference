#!/usr/bin/env bash
set -euo pipefail
source "$PROJECT_ROOT/scripts/e007/env_arc.sh"
mkdir "$R"
scontrol show job "$SLURM_JOB_ID" > "$R/allocation.txt"
grep -q 'UserId=hpuvvad' "$R/allocation.txt"
grep -q 'JobState=RUNNING' "$R/allocation.txt"
grep -q 'NodeList=c58' "$R/allocation.txt"
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c '^c58$' "$R/launcher.log") == 1 ]]
(cd "$E006_CONTROL"; sha256sum -c results/e006_20261106/handoff.sha256; sha256sum -c results/e006_20261106/final-binaries.sha256) > "$R/e006-preservation-check.log"
{
 date --iso-8601=seconds
 lscpu
 numactl --hardware
 cat /proc/self/status
 gcc --version
 mpirun --version
 nvcc --version
 compute-sanitizer --version
 ncu --version
 nvidia-smi -q
} > "$R/metadata.txt" 2>&1
run runtime "$E005_PYTHON" scripts/e004/check_runtime.py
run retained-project bash /home/hpuvvad/csc548-w4a16-e004-20261106/scripts/check_inblock.sh
"$E005_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u "$E006_CONTROL/../csc548-w4a16-e004-20261106/results/e004_20261106/pip-freeze.txt" "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
"$E005_PYTHON" scripts/e007/prepare_source.py "$E006_CONTROL/external/patched" external > "$R/prepare.log"
cp external/e007-lookahead.patch external/buffer-model.json "$R/"
TORCH_ROOT=$("$E005_PYTHON" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
flags=(-DUSE_CUDA -O3 -DNDEBUG -std=c++17 -lineinfo -arch=sm_120 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -Xptxas=-v)
for variant in control candidate; do
 mkdir "$R/$variant"
 run "build-$variant" nvcc "${flags[@]}" -I"external/$variant/csrc/quantization/marlin" -I"external/$variant/csrc" \
  -I"$TORCH_ROOT/include" -I"$TORCH_ROOT/include/torch/csrc/api/include" \
  scripts/e005/local.cu -L"$TORCH_ROOT/lib" -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
  -Xlinker=-rpath -Xlinker="$TORCH_ROOT/lib" -o "$R/$variant/libe005.so"
 sha256sum "$R/$variant/libe005.so" scripts/e005/local.cu external/"$variant"/csrc/quantization/marlin/* \
  external/"$variant"/csrc/core/scalar_type.hpp > "$R/$variant/build-source.sha256"
 ldd "$R/$variant/libe005.so" > "$R/$variant/ldd.txt"
 cuobjdump --dump-resource-usage "$R/$variant/libe005.so" > "$R/$variant/resources.txt"
 cuobjdump --dump-sass "$R/$variant/libe005.so" | gzip > "$R/$variant/sass.txt.gz"
done
tar -czf "$R/matched-source.tar.gz" external
sha256sum scripts/e007/* scripts/e005/bench.py scripts/e004/bench.py > "$R/build-tooling.sha256"
