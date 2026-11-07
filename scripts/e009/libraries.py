#!/usr/bin/env python3
"""Hash the actual mapped CUDA/cuBLAS/adapter binaries from prerequisite logs."""
import hashlib
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = set()
for name in ('runtime.log','smoke.log'):
    for line in (root/name).read_text().splitlines():
        fields = line.split()
        if fields and fields[-1].startswith('/') and any(token in Path(fields[-1]).name for token in
                ('libcublas','libcuda','libe004','libe005')):
            paths.add(Path(fields[-1]))
assert any(p.name.startswith('libcublas.so') for p in paths)
assert any(p.name.startswith('libcublasLt.so') for p in paths)
assert any(p.name.startswith('libcuda.so') for p in paths)
assert any(p.name.startswith('libcudart.so') for p in paths)
output = root/'mapped-libraries.sha256'
assert not output.exists()
with output.open('x') as out:
    for path in sorted(paths):
        with path.open('rb') as binary:
            digest = hashlib.file_digest(binary,'sha256').hexdigest()
        out.write(digest+'  '+str(path)+'\n')
print('PASS: actual mapped library identity recorded',len(paths),flush=True)
