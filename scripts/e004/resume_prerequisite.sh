#!/usr/bin/env bash
# E004's first bridge binary compiled, then a directory in the hash glob failed.
# Preserve that diagnostic and re-hash the unchanged binary with file-only globs.
set -euo pipefail
: "${PROJECT_ROOT:?Absolute isolated project}"
: "${E004_PYTHON:?Isolated Python}"
: "${SLURM_JOB_ID:?Verified allocation}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR="$PROJECT_ROOT/results/e004_20261106"
mv "$RUN_DIR/bridge-build.log" "$RUN_DIR/bridge-build-attempt1.log"
mv "$RUN_DIR/bridge-source.sha256" "$RUN_DIR/bridge-source-attempt1.sha256"
sha256sum scripts/e004/*.py scripts/e004/*.cu scripts/e004/*.sh src/*.cpp src/cuda/* include/w4a16/* "$RUN_DIR/libe004.so" > "$RUN_DIR/bridge-source.sha256"
prun bash scripts/check_inblock.sh > "$RUN_DIR/project-check.log" 2>&1
"$E004_PYTHON" -m pip install --no-deps 'https://files.pythonhosted.org/packages/8b/a8/92e278093df40fde4c4bfd6572704bb8f398d6a2280d8c82cb5efaf12047/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa' > "$RUN_DIR/pip-vllm.log" 2>&1
"$E004_PYTHON" -m pip freeze > "$RUN_DIR/pip-freeze.txt"
prun "$E004_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_arch_list()); print(torch.__config__.show()); a=torch.ones((16,16),device="cuda"); print((a@a).sum().item())' > "$RUN_DIR/torch-smoke.log" 2>&1
