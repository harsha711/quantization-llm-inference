#!/usr/bin/env python3
"""Inspect prerequisite evidence before dependent jobs are submitted."""
import argparse
import csv
import json
from pathlib import Path
import re


def blocks(path):
    out, name = {}, None
    for line in path.read_text().splitlines():
        if 'Function :' in line:
            name = re.sub(r'_GLOBAL__N__[0-9a-f]+_', '_GLOBAL__N__', line.split()[-1])
            out[name] = []
        elif name and re.match(r'\s*/\*([0-9a-f]{4,}\*/| 0x)', line):
            out[name].append(line.strip())
    return out


def inspect(root, stage, baseline=None):
    assert (root/'driver.exit').read_text().strip() == '0'
    assert not (root/'dependency-diff.txt').read_text()
    assert (root/'e010-inputs-check.log').read_text().count(': OK') == 5
    allocation = (root/'allocation.txt').read_text()
    assert 'UserId=hpuvvad' in allocation and 'JobState=RUNNING' in allocation
    for tag in ('project-build','project-ctest','project-gpu','v3-check','library-build','smoke'):
        assert (root/(tag+'.exit')).read_text().strip() == '0', tag
    for tag in ('project-gpu','v3-check'):
        log = (root/(tag+'.log')).read_text()
        assert 'ERROR SUMMARY: 0 errors' in log and 'RACECHECK SUMMARY: 0 hazards' in log, tag
        assert 'NOT run' not in log and 'unavailable' not in log, tag
    bench, lib = blocks(root/'sass-bench.txt'), blocks(root/'sass-library.txt')
    selected = {k:v for k,v in lib.items() if 'w4a16_v3_kernel' in k or 'w4a16_tc_kernel' in k}
    assert selected and all(k in bench and v == bench[k] for k,v in selected.items()), 'bench/library SASS mismatch'
    rows = [json.loads(x) for x in (root/'smoke.jsonl').read_text().splitlines()]
    expected = {'v3','tc_ladder','marlin','cublas_dq'} | ({'v3_m2'} if stage == 'validation' else set())
    assert {r['config'] for r in rows} == expected and len(rows) == len(expected)
    for row in rows:
        assert row['traced_kernels'] and row['iters'] == 0
        assert all(v['num_bad'] == 0 and v['checked'] == row['M']*row['N'] for v in row['validation'])
    if stage == 'baseline':
        assert (root/'calibration.exit').read_text().strip() == '0'
        text = (root/'calibration.raw.csv').read_text()
        assert 'elementwise_kernel' in text and 'add' in text, 'profile window did not select final add'
        assert (root/'baseline-v3.exit').read_text().strip() == '0'
        rows = [json.loads(x) for x in (root/'ncu/baseline-v3.jsonl').read_text().splitlines()]
        assert rows and all(r['launch_skip'] == 399 and r['post_profile_validation'] == r['validation'] for r in rows)
        raw = list(csv.reader(line for line in (root/'ncu/baseline-v3.raw.csv').read_text().splitlines() if not line.startswith('==')))
        assert len(raw) == 3 and len(raw[0]) == len(raw[2])
        values = dict(zip(raw[0],raw[2]))
        metrics = {m:values[m] for m in Path(__file__).with_name('metrics.txt').read_text().splitlines()}
        assert metrics.get('smsp__inst_executed.sum') is not None
        print('BASELINE PASS', json.dumps(metrics,sort_keys=True))
    else:
        for tag in ('candidate-check','validation','cycle-memcheck','cycle-racecheck','cycle-synccheck'):
            assert (root/(tag+'.exit')).read_text().strip() == '0', tag
        rows = [json.loads(x) for x in (root/'validation.jsonl').read_text().splitlines()]
        assert len(rows) == 75 and all(v['num_bad'] == 0 and v['checked'] == r['M']*r['N'] for r in rows for v in r['validation'])
        if baseline:
            old = blocks(baseline/'sass-library.txt')
            for k,v in selected.items():
                assert old[k] == v, 'retained control device code changed'
        print('VALIDATION PASS: complete all-row/parent-bitwise checks, sanitizers and retained-control SASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--stage',choices=('baseline','validation'),required=True)
    p.add_argument('--baseline',type=Path)
    args = p.parse_args()
    inspect(args.root,args.stage,args.baseline)
