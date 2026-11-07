#!/usr/bin/env python3
"""Accept full real evidence, reject missing rows, aliases and operand mismatch."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).resolve()
analyzer = Path(__file__).with_name('analyze.py')
with tempfile.TemporaryDirectory(prefix='e008-negative-') as directory:
    root = Path(directory)/'evidence'
    shutil.copytree(source,root)
    def check(expected):
        result = subprocess.run([sys.executable,str(analyzer),str(root)],capture_output=True,text=True)
        assert (result.returncode == 0) == expected, (result.stdout,result.stderr)
    check(True)
    path = root/'series2.jsonl'
    original = path.read_text()
    path.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    check(False)
    path.write_text(original)
    for mutation in ('alias','operand'):
        rows = [json.loads(x) for x in original.splitlines()]
        if mutation == 'alias':
            rows[0]['allocations'][1]['packed_range'] = rows[0]['allocations'][0]['packed_range']
        else:
            rows[0]['allocations'][0]['operand_sha256']['Q'] = '0'*64
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        check(False)
    path.write_text(original)
    check(True)
print('PASS: full evidence accepted, missing timing row, aliased weights and mismatched operands rejected')
