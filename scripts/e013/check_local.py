#!/usr/bin/env python3
"""Offline E013 source, traversal and corruption checks. No GPU claims."""
import copy
import json
from pathlib import Path
import tempfile

import analyze
import prepare_source


def rejects(function):
    try:
        function()
    except (AssertionError, FileExistsError):
        return
    raise AssertionError('invalid input accepted')


root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='e013-source-check-') as tmp:
    out = Path(tmp)/'source'
    archive = root/'results/e006_20261106/matched-source.tar.gz'
    prepare_source.prepare(archive,out)
    host = (out/'csrc/quantization/marlin/marlin.cu').read_text()
    assert host.count(prepare_source.NEW) == 1
    assert 'thread_tfg = {64, 128, 128};' in host
    rejects(lambda: prepare_source.prepare(archive,out))
    bad = Path(tmp)/'bad.tar.gz'
    bad.write_bytes(archive.read_bytes()+b'corruption')
    rejects(lambda: prepare_source.prepare(bad,Path(tmp)/'bad'))
    assert not (Path(tmp)/'bad').exists()

first = [(k,n,p,c) for k,n,p,c,_ in analyze.traversal(analyze.SHAPES,1)]
second = [(k,n,p,c) for k,n,p,c,_ in analyze.traversal(analyze.SHAPES[::-1],2)]
assert first[::-1] == second and len(set(first)) == 4
assert analyze.passes([5,6]) and not analyze.passes([5,4.99])
assert not analyze.passes([6,-1]) and not analyze.passes([6,float('nan')])
assert not analyze.passes([6])
assert analyze.gain(100,90) > 9.99
rejects(lambda: analyze.gain(0,90))

# Exercise the new checker using a valid historical row, without fabricating GPU evidence.
rows = [json.loads(x) for x in (root/'results/e010_c4_20261107/series1.jsonl').read_text().splitlines()]
row = copy.deepcopy(next(v for v in rows if v['config'] == 'marlin' and v['M'] == 128))
row.update(config='A',pattern='mixed',activation_sha256=row['activation_prefix_sha256'],
    identities=dict(local='a'*64,bridge='85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f',
                    wheel='e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'),
    runtime=dict(host='c4',threads=16,torch='2.14.0+cu130',cuda='13.0',python='3.11.0',gpu='GPU-offline-test'))
analyze.checked(row,4,[3,4,5,6],timed=True)
for mutation in ('median','missing_sample','nonfinite','alias','output','tile','guard','seed','runtime'):
    bad = copy.deepcopy(row)
    if mutation == 'median': bad['median_us'] += 1
    elif mutation == 'missing_sample': bad['samples_us'].pop()
    elif mutation == 'nonfinite': bad['samples_us'][0] = float('nan')
    elif mutation == 'alias': bad['allocations'][1]['packed_range'] = bad['allocations'][0]['packed_range']
    elif mutation == 'output': bad['validation'][0]['num_bad'] = 1
    elif mutation == 'tile': bad['resources'][0]['thread_n'] = 128
    elif mutation == 'guard': bad['workspace_guards'] = False
    elif mutation == 'seed': bad['seeds'][0] = 7
    elif mutation == 'runtime': bad['runtime']['gpu'] = 'unknown'
    rejects(lambda: analyze.checked(bad,4,[3,4,5,6],timed=True))
print('PASS: exact source extraction/overwrite protection, reversed traversal, fixed 5% gate and nine evidence corruptions')
