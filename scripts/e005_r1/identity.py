#!/usr/bin/env python3
"""Verify the exact E006 body/evidence and current one-rank target runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

root = Path(os.environ['PROJECT_ROOT'])
e006 = Path(os.environ['E006_CONTROL'])
r = e006/'results/e006_20261106'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checked = {}
for line in (r/'patched/build-source.sha256').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    if name.startswith('external/patched/') or name.endswith('libe005.so') or name == 'scripts/e005/local.cu':
        path = e006/name
        assert sha(path) == expected, path
        checked[str(path)] = expected
for name in ('scripts/e004/bench.py', 'scripts/e004/check_runtime.py', 'scripts/e005/bench.py', 'scripts/e005/local.cu'):
    assert sha(root/name) == sha(e006/name), name
    checked[name] = sha(root/name)
assert sha(Path(os.environ['E005_LOCAL'])) == 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
assert sha(Path(os.environ['E005_BRIDGE'])) == '85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f'
assert sha(Path(os.environ['E005_WHEEL'])) == 'e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'
evidence = json.loads((r/'analysis.json').read_text())
assert evidence['binary_sha256']['patched'] == sha(Path(os.environ['E005_LOCAL']))
assert sum(v['rows'] for k,v in evidence['matrix'].items() if k.startswith('validation')) == 135
assert sum(v['rows'] for k,v in evidence['matrix'].items() if not k.startswith('validation')) == 162
assert not (r/'VALIDATED').exists()
assert not (Path(os.environ['E005_CONTROL'])/'results/e005_20261106/VALIDATED').exists()
assert os.environ['OMPI_COMM_WORLD_SIZE'] == '1' and os.environ['OMPI_COMM_WORLD_RANK'] == '0'
assert torch.get_num_threads() == 16
assert torch.__version__ == '2.14.0+cu130' and torch.version.cuda == '13.0'
assert torch.cuda.get_device_capability() == (12,0)
gpu = torch.cuda.get_device_properties(0)
assert gpu.multi_processor_count == 36 and gpu.name == 'NVIDIA GeForce RTX 5060 Ti'
device = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()
assert device == 'GPU-51745a0b-32d9-31c9-1ecd-359ac0ac89cc, 615.71.09', device
for cmd, version in [('nvcc','V13.4.92'),('compute-sanitizer','2026.3.0.0'),('ncu','2026.3.1.0')]:
    assert version in subprocess.check_output([cmd,'--version'],text=True)
print(json.dumps(dict(status='identity verified, E006 validation reused', executable=sys.executable,
    torch=torch.__version__,cuda=torch.version.cuda,device=device,threads=torch.get_num_threads(),
    affinity=sorted(os.sched_getaffinity(0)),checked=checked,e006_analysis_sha256=sha(r/'analysis.json')),
    indent=2),flush=True)
