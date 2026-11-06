#!/usr/bin/env python3
"""Cheap target-runtime check, executed as a file because prun joins arguments."""
import hashlib
from pathlib import Path
import sysconfig
import torch

print(torch.__version__, torch.version.cuda, flush=True)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list(), flush=True)
print(torch.__config__.show(), flush=True)
a = torch.ones((16,16),device="cuda")
assert (a@a).sum().item() == 4096.0
lib=Path(sysconfig.get_paths()["purelib"])/"vllm/_C_stable_libtorch.abi3.so"
print("LIBRARY",lib,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
torch.ops.load_library(str(lib))
print(torch.ops._C.marlin_gemm.default._schema,flush=True)
print(torch.ops._C.gptq_marlin_repack.default._schema,flush=True)
print(Path('/proc/self/maps').read_text(),flush=True)
