#!/usr/bin/env bash
# Run via ssh on the already-allocated node; this is not a Slurm batch script.
set -euo pipefail
: "${PROJECT_ROOT:?Absolute E004 shared project root}"
: "${SLURM_JOB_ID:?Verify allocation first}"
: "${E004_PYTHON:?Absolute isolated Python path}"
cd "$PROJECT_ROOT"
export PATH=/usr/local/cuda/bin:$PATH
export SLURM_NTASKS=1 SLURM_TASKS_PER_NODE=1 SLURM_JOB_CPUS_PER_NODE=16 SLURM_CPUS_PER_TASK=16
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true
export OMPI_MCA_rmaps_base_mapping_policy=ppr:1:node OMPI_MCA_hwloc_base_binding_policy=none
export RUN_DIR="$PROJECT_ROOT/results/e004_20261106"
mkdir -p "$RUN_DIR"
{
  date -Is; hostname; scontrol show job "$SLURM_JOB_ID"
  env | sort | grep -E '^(SLURM|OMP|OMPI|CUDA|LMOD_FAMILY)' || true
  lscpu; lscpu -e; cat /proc/cpuinfo; numactl --hardware; taskset -pc $$
  findmnt -T "$PROJECT_ROOT"; getconf GNU_LIBC_VERSION
  nvidia-smi -q; nvidia-smi topo -m
  nvcc --version; c++ --version; mpirun --version; cmake --version
  ncu --version; compute-sanitizer --version
} > "$RUN_DIR/metadata.txt" 2>&1
prun /bin/hostname > "$RUN_DIR/launcher.log" 2>&1
mkdir -p external/vllm
[[ -f external/vllm/LICENSE ]] || tar -xzf e004-marlin-source.tar.gz -C external/vllm
prun bash scripts/build.sh > "$RUN_DIR/project-build.log" 2>&1
prun bash scripts/e004/build_bridge.sh > "$RUN_DIR/bridge-build.log" 2>&1
prun bash scripts/check_inblock.sh > "$RUN_DIR/project-check.log" 2>&1
"$E004_PYTHON" -m pip install --no-deps 'https://files.pythonhosted.org/packages/8b/a8/92e278093df40fde4c4bfd6572704bb8f398d6a2280d8c82cb5efaf12047/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa' > "$RUN_DIR/pip-vllm.log" 2>&1
"$E004_PYTHON" -m pip freeze > "$RUN_DIR/pip-freeze.txt"
prun "$E004_PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_arch_list()); print(torch.__config__.show()); a=torch.ones((16,16),device="cuda"); print((a@a).sum().item())' > "$RUN_DIR/torch-smoke.log" 2>&1
