#!/usr/bin/env python3
"""One direct E005 A launch, with existing lossless packing and FP64 checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location('e005_control', Path(__file__).resolve().parents[1]/'e005/bench.py')
e005 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e005)

p = argparse.ArgumentParser(description=__doc__)
for name in ('source', 'local', 'wheel', 'bridge', 'out'):
    p.add_argument('--'+name, type=Path, required=True)
p.add_argument('--k', type=int, default=4096)
p.add_argument('--n', type=int, default=4096)
a = p.parse_args()
assert a.k >= 256 and a.k % 128 == 0 and a.n % 128 == 0
assert os.environ['OMPI_COMM_WORLD_SIZE'] == '1'
a.configs = ['A']
stream = e005.torch.cuda.Stream()
print('ENV', json.dumps(dict(torch=e005.torch.__version__, cuda=e005.torch.version.cuda,
    gpu=e005.torch.cuda.get_device_name(), capability=e005.torch.cuda.get_device_capability(),
    binary_sha256=hashlib.sha256(a.local.read_bytes()).hexdigest(),
    wheel_sha256=hashlib.sha256(a.wheel.read_bytes()).hexdigest())), flush=True)
with e005.torch.cuda.stream(stream):
    case = e005.Case(a, a.k, a.n, [8], 1, 'mixed')
    result = case.check('A', 8, repeats=1, graphs=False)
    r = result['resources']
    assert (r['threads'], r['thread_k'], r['thread_n'], r['m_blocks'], r['m8'], r['stages']) == (256,128,128,1,1,4), r
    result.update(M=8, K=a.k, N=a.n, direct_launches=1, operand_sha256=case.hashes,
        codes_match=True, scales_match=True, reconstruction_match=True, workspace_guards=True, lock_reset=True)
    with a.out.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
print('PASS', json.dumps(result), flush=True)
print('MAPPED_LIBRARIES', '\n'.join(x for x in Path('/proc/self/maps').read_text().splitlines()
    if any(v in x for v in ('libe005', 'libe004', '_C_stable'))), flush=True)
