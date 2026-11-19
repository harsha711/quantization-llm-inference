#!/usr/bin/env python3
"""Reject corrupt real E053 samples and parent-output identities using temporary copies."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import analyze

source = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix='e053-negative-') as directory:
    root = Path(directory)
    for name in ('series1.jsonl','series2.jsonl','single-s1.csv','single-s2.csv'):
        shutil.copyfile(source/name,root/name)
    analyze.analyze(root)
    path = root/'series2.jsonl'
    original = path.read_text()
    rows = [json.loads(x) for x in original.splitlines()]
    mutations = []
    mutations.append(rows[:-1])
    for kind in ('sample','median','parent','operand'):
        bad = copy.deepcopy(rows)
        if kind == 'sample': bad[0]['samples_us'][0] = -1
        elif kind == 'median': bad[0]['median_us'] *= 2
        elif kind == 'parent':
            next(r for r in bad if r['config'] == 'v3_m2')['validation'][0]['output_sha256'] = 'f'*64
        else: bad[0]['allocations'][0]['operand_sha256']['Q'] = 'f'*64
        mutations.append(bad)
    for bad in mutations:
        path.write_text(''.join(json.dumps(r)+'\n' for r in bad))
        try:
            analyze.analyze(root)
        except AssertionError:
            continue
        raise AssertionError('damaged evidence accepted')
    path.write_text(original)
    analyze.analyze(root)
print('PASS: real evidence accepted; missing rows, negative samples, changed medians/parent bits/operands rejected')
