#!/usr/bin/env bash
# Appended after the local staging payload in the submitted absolute-path job.
source "$PROJECT_ROOT/scripts/e017/env_arc.sh"
exec > "$R/bootstrap-driver.log" 2>&1
trap 'rc=$?;echo "$rc" > "$R/bootstrap-driver.exit";touch "$R/BOOTSTRAP_FINISHED";for ((i=0;i<300;i++));do [[ -e "$R/BOOTSTRAP_RETRIEVED" ]] && break;sleep 2;done;exit "$rc"' EXIT
allocation bootstrap
prun /bin/hostname > "$R/launcher.log" 2>&1
[[ $(grep -c "^$SLURM_JOB_NODELIST$" "$R/launcher.log") == 1 ]]
{
 date --iso-8601=seconds;hostname;lscpu;lscpu -e;cat /proc/cpuinfo;numactl --hardware;cat /proc/self/status
 gcc --version;mpirun --version;nvcc --version;compute-sanitizer --version
 nvidia-smi -q;nvidia-smi topo -m;findmnt -T "$PROJECT_ROOT";df -h "$PROJECT_ROOT"
 env | grep -E '^(SLURM_|OMP_|OMPI_|PATH=|PROJECT_ROOT=)' | sort
} > "$R/metadata.txt" 2>&1
clock > "$R/initial-clocks.csv"
export PIP_CACHE_DIR=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/cache/pip
if [[ ! -x "$E017_PYTHON" ]];then
 run venv python3.11 -m venv /tmp/e017-hpuvvad-20261109
 run install "$E017_PYTHON" -m pip install -r scripts/e014/requirements.lock
fi
"$E017_PYTHON" -m pip freeze > "$R/pip-freeze.txt"
diff -u scripts/e014/requirements.lock "$R/pip-freeze.txt" > "$R/dependency-diff.txt"
mkdir -p external
# Reuse an exact already-installed official packer when available on this node.
for candidate in /mnt/beegfs/hpuvvad/csc548-w4a16-e017-20261109/external/_C_stable_libtorch.abi3.so /tmp/e004-hpuvvad-20261106/lib/python3.11/site-packages/vllm/_C_stable_libtorch.abi3.so /tmp/e005-r2-hpuvvad-20261106/lib/python3.11/site-packages/vllm/_C_stable_libtorch.abi3.so;do
 if [[ -f "$candidate" ]];then cp "$candidate" "$E017_WHEEL";break;fi
done
if [[ ! -f "$E017_WHEEL" ]];then
 run packer-download "$E017_PYTHON" -m pip download --no-deps --dest external 'https://files.pythonhosted.org/packages/8b/a8/92e278093df40fde4c4bfd6572704bb8f398d6a2280d8c82cb5efaf12047/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa'
 run extract-packer "$E017_PYTHON" scripts/e017/extract_packer.py
fi
sha256sum "$E017_LOCAL" "$E017_WHEEL" "$E017_BRIDGE" > "$R/binaries.sha256"
grep -q '^e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7 ' "$R/binaries.sha256"
grep -q '^e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9 ' "$R/binaries.sha256"
grep -q '^85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f ' "$R/binaries.sha256"
find src include scripts/e017 scripts/e014 scripts/e004 scripts/e005 -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > "$R/staged-source.sha256"
[[ $(df --output=avail -B1 "$PROJECT_ROOT" | tail -1) -gt 8589934592 ]]
dd if=/dev/zero of="$R/storage-probe.bin" bs=1048576 count=1
sha256sum "$R/storage-probe.bin" > "$R/storage-probe.sha256"
run runtime "$E017_PYTHON" scripts/e016/runtime.py "$R"
run retained-build bash scripts/build.sh
run retained-ctest ctest --test-dir build --output-on-failure
run retained-gpu-check bash scripts/check_inblock.sh
bash scripts/e017/prerequisite_arc.sh
