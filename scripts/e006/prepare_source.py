#!/usr/bin/env python3
"""Generate matched E005 control and exactly one E006 device-body correction."""
import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('out', type=Path)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
control = a.out/'control'
patched = a.out/'patched'
subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]/'e005/prepare_source.py'), str(a.source), str(control)], check=True)
shutil.copytree(control, patched)
f = patched/'csrc/quantization/marlin/marlin_template.h'
before = f.read_text()
old = '      cp_async_wait<0>();\n      bool last = slice_idx == slice_count - 1;'
new = '      cp_async_wait<0>();\n      // Join all shared readers and drained copy groups before overlay reuse.\n      __syncthreads();\n      bool last = slice_idx == slice_count - 1;'
assert before.count(old) == 1
after = before.replace(old, new)
assert after.count('__syncthreads();') == before.count('__syncthreads();')+1
assert after.replace(new, old) == before
f.write_text(after)
(a.out/'e006-barrier.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/csrc/quantization/marlin/marlin_template.h', tofile='b/csrc/quantization/marlin/marlin_template.h')))
for c in control.rglob('*'):
    if c.is_file() and c.name != 'marlin_template.h':
        assert c.read_bytes() == (patched/c.relative_to(control)).read_bytes(), c
assert (control/'csrc/quantization/marlin/marlin_template.h').read_bytes() == (a.source/'csrc/quantization/marlin/marlin_template.h').read_bytes()
print('PASS: E005 host policy preserved, one block-wide barrier added, all other files identical.')
