#!/usr/bin/env python3
"""Cheap target-runtime check, executed as a file because prun joins arguments."""
import hashlib
import os
import socket
from pathlib import Path
import sysconfig
import torch

print("RANK_ENVIRONMENT",socket.gethostname(),os.getpid(),sorted(os.sched_getaffinity(0)),{k:v for k,v in os.environ.items() if k.startswith(("OMP_","OMPI_COMM_WORLD_","SLURM_"))},flush=True)
print("TORCH_THREADS",torch.get_num_threads(),torch.get_num_interop_threads(),flush=True)
print(torch.__version__, torch.version.cuda, flush=True)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list(), flush=True)
print(torch.__config__.show(), flush=True)
assert torch.ones((1024,1024)).sum().item() == 1048576.0
for status in sorted(Path('/proc/self/task').glob('*/status')):
    details=[x for x in status.read_text().splitlines() if x.startswith(('Name:', 'Cpus_allowed_list:'))]
    print("THREAD_AFFINITY",status.parent.name,details,flush=True)
a = torch.ones((16,16),device="cuda")
assert (a@a).sum().item() == 4096.0
lib=Path(sysconfig.get_paths()["purelib"])/"vllm/_C_stable_libtorch.abi3.so"
print("LIBRARY",lib,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
torch.ops.load_library(str(lib))
print(torch.ops._C.marlin_gemm.default._schema,flush=True)
print(torch.ops._C.gptq_marlin_repack.default._schema,flush=True)
print(Path('/proc/self/maps').read_text(),flush=True)
