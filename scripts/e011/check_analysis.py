#!/usr/bin/env python3
"""Check the complete E011 analyzer, including deliberate corruptions of real evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).resolve()
analyzer = Path(__file__).with_name('analyze.py')
with tempfile.TemporaryDirectory(prefix='e011-negative-') as directory:
    root = Path(directory) / 'evidence'
    shutil.copytree(source, root)

    def check(expected):
        result = subprocess.run([sys.executable, str(analyzer), str(root)], capture_output=True, text=True)
        assert (result.returncode == 0) == expected, (result.stdout, result.stderr)

    check(True)
    path = root / 'series2.jsonl'
    original = path.read_text()
    for mutation in ('missing_row', 'operand', 'alias', 'mode', 'statistics'):
        rows = [json.loads(line) for line in original.splitlines()]
        if mutation == 'missing_row':
            rows.pop()
        elif mutation == 'operand':
            rows[0]['activation_prefix_sha256'] = '0' * 64
        elif mutation == 'alias':
            rows[0]['allocations'][1]['packed_range'] = rows[0]['allocations'][0]['packed_range']
        elif mutation == 'mode':
            rows[0]['cublas_math_mode'] = 0
        else:
            rows[0]['median_us'] *= 2
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        check(False)
    path.write_text(original)
    trace = root / 'trace_marlin_M4_K4096_N11008.json'
    original = trace.read_text()
    row = json.loads(original)
    row['launch_skip'] -= 4
    trace.write_text(json.dumps(row))
    check(False)
    trace.write_text(original)
    metric = root / 'ncu/marlin_M4_K4096_N11008.csv'
    metric.rename(metric.with_suffix('.held'))
    check(False)
    metric.with_suffix('.held').rename(metric)
    check(True)
print('PASS: complete evidence accepted. Missing rows/profiles, wrong operands, aliases, mode, statistics and target position rejected')
