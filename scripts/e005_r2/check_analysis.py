#!/usr/bin/env python3
"""Require the evidence checkers to reject incomplete and wrong-kernel records."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source=Path(sys.argv[1]).resolve()
scripts=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='e005-r2-negative-') as d:
    r=Path(d)/'evidence';shutil.copytree(source,r)
    def check(script,success):
        result=subprocess.run([sys.executable,str(scripts/script),str(r)],text=True,capture_output=True)
        assert (result.returncode==0)==success,(script,result.stdout,result.stderr)
    check('analyze.py',True)
    check('analyze_profiles.py',True)
    p=r/'series2.jsonl';original=p.read_text()
    p.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    check('analyze.py',False)
    p.write_text(original)
    check('analyze.py',True)
    shape=json.loads((r/'analysis.json').read_text())['profile_shape']
    p=r/f'ncu/A_M4_K{shape[0]}_N{shape[1]}_reused_kernel_replay_mangled.csv'
    rows=list(csv.reader(p.open()));rows[2][rows[0].index('Kernel Name')]='unrelated_kernel'
    with p.open('w',newline='') as f:csv.writer(f).writerows(rows)
    check('analyze_profiles.py',False)
print('PASS: complete timing/profile evidence accepted, missing timing row and wrong kernel rejected')
