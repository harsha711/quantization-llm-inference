#!/usr/bin/env python3
"""Accept real E013 evidence and reject corrupted copies without GPU reruns."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).resolve()
analyzer = Path(__file__).with_name('analyze.py')
extra = sys.argv[2:]
with tempfile.TemporaryDirectory(prefix='e013-negative-') as tmp:
    root = Path(tmp)/'evidence'
    shutil.copytree(source,root)
    def check(expected):
        result = subprocess.run([sys.executable,str(analyzer),str(root),*extra],capture_output=True,text=True)
        assert (result.returncode == 0) == expected,(result.stdout,result.stderr)
    check(True)
    path = root/'series2.jsonl'
    original = path.read_text()
    for mutation in ('missing_row','operand','tile','median','sample','runtime'):
        rows = [json.loads(x) for x in original.splitlines()]
        if mutation == 'missing_row': rows.pop()
        elif mutation == 'operand': rows[0]['allocations'][0]['packed_sha256'] = '0'*64
        elif mutation == 'tile': rows[0]['resources'][0]['thread_n'] = 64
        elif mutation == 'median': rows[0]['median_us'] += 1
        elif mutation == 'sample': rows[0]['samples_us'].pop()
        elif mutation == 'runtime': rows[0]['runtime']['gpu'] = 'GPU-another-device'
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        check(False)
    path.write_text(original)
    log = root/'racecheck.log'
    original_log = log.read_text()
    log.write_text(original_log.replace('0 errors, 0 warnings','1 errors, 0 warnings'))
    check(False)
    log.write_text(original_log)
    check(True)
print('PASS: complete evidence accepted, seven damaged-data cases rejected')
