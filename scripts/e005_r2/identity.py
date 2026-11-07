#!/usr/bin/env python3
"""Verify preserved inputs on the changed node, without reusing its old gate."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import torch

root=Path(os.environ['PROJECT_ROOT'])
e006=Path(os.environ['E006_CONTROL'])
r=e006/'results/e006_20261106'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checked={}
for line in (r/'patched/build-source.sha256').read_text().splitlines():
    expected,name=line.split(maxsplit=1)
    if name.startswith('external/patched/') or name.endswith('libe005.so') or name=='scripts/e005/local.cu':
        assert sha(e006/name)==expected,name
        checked[str(e006/name)]=expected
for name in ('scripts/e004/bench.py','scripts/e004/check_runtime.py','scripts/e005/bench.py','scripts/e005/local.cu'):
    assert sha(root/name)==sha(e006/name),name
    checked[name]=sha(root/name)
for key,expected in {
    'E005_LOCAL':'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7',
    'E005_BRIDGE':'85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f',
    'E005_WHEEL':'e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9',
}.items():
    path=Path(os.environ[key]);assert sha(path)==expected,key
    checked[str(path)]=expected
assert not (r/'VALIDATED').exists()
assert not (Path(os.environ['E005_CONTROL'])/'results/e005_20261106/VALIDATED').exists()
assert socket.gethostname()=='c10'
assert os.environ['OMPI_COMM_WORLD_SIZE']=='1' and os.environ['OMPI_COMM_WORLD_RANK']=='0'
assert torch.get_num_threads()==16 and torch.__version__=='2.14.0+cu130' and torch.version.cuda=='13.0'
gpu=torch.cuda.get_device_properties(0)
assert torch.cuda.get_device_capability()==(12,0) and gpu.multi_processor_count==36
assert gpu.name=='NVIDIA GeForce RTX 5060 Ti'
device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()
assert device.endswith(', 610.57.04'),device
for cmd,version in [('nvcc','V13.4.92'),('compute-sanitizer','2026.3.0.0'),('ncu','2026.3.1.0')]:
    assert version in subprocess.check_output([cmd,'--version'],text=True)
print(json.dumps(dict(status='preserved source/binaries match, fresh c10 validation required',
    python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,device=device,
    threads=torch.get_num_threads(),affinity=sorted(os.sched_getaffinity(0)),checked=checked),indent=2))
