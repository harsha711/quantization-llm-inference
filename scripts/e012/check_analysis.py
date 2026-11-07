#!/usr/bin/env python3
"""Accept the real E012 evidence and reject damaged records without GPU reruns."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

import analyze
from regions import verified_sources

root, old, e006 = (Path(p) for p in sys.argv[1:])
analyze.main(root, old, e006)
sources = verified_sources(e006)
tag = 'instruction_s1_M4_K4096_N11008'
trace = 'trace_marlin_M4_K4096_N11008.json'
paths = [Path(trace), Path(tag+'.log'), Path(tag+'.exit'), Path('export_'+tag+'.exit')]
paths += [Path('ncu')/(tag+suffix) for suffix in ('.ncu-repz', '.raw.csv', '.source.csv', '.pc.json.gz')]
with tempfile.TemporaryDirectory(prefix='e012-negative-') as temp:
    target = Path(temp)
    for name in paths:
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target/name)
    pcpath = target/'ncu'/(tag+'.pc.json.gz')
    original = pcpath.read_bytes()
    data = json.loads(gzip.decompress(original))
    for mutation in ('duplicate_pc', 'counter_sum', 'phase', 'source_hash', 'template'):
        changed = copy.deepcopy(data)
        if mutation == 'duplicate_pc':
            changed['pcs'].append(copy.deepcopy(changed['pcs'][0]))
        elif mutation == 'counter_sum':
            changed['pcs'][0]['counts']['inst_executed'] += 1
        elif mutation == 'phase':
            changed['pcs'][0]['region'] = 'epilogue'
        elif mutation == 'source_hash':
            changed['source_sha256'][next(iter(changed['source_sha256']))] = '0'*64
        else:
            prefix, remaining = changed['name'].split('<', 1)
            arguments, suffix = remaining.split('>', 1)
            values = arguments.split(','); values[4] = '128'
            changed['name'] = prefix+'<'+','.join(values)+'>'+suffix
        pcpath.write_bytes(gzip.compress(json.dumps(changed).encode()))
        try:
            analyze.profile_checked(target,'instruction',1,4,4096,11008,sources)
        except AssertionError:
            pass
        else:
            raise AssertionError(mutation)
        pcpath.write_bytes(original)
    for filename, replace in [(tag+'.log', lambda s:s.replace('--launch-skip 399','--launch-skip 0')),
                              (tag+'.exit', lambda s:'99\n')]:
        p=target/filename; original_text=p.read_text(); p.write_text(replace(original_text))
        try:
            analyze.profile_checked(target,'instruction',1,4,4096,11008,sources)
        except AssertionError:
            pass
        else:
            raise AssertionError(filename)
        p.write_text(original_text)
    (target/'ncu'/(tag+'.ncu-repz')).unlink()
    try:
        analyze.profile_checked(target,'instruction',1,4,4096,11008,sources)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('missing report accepted')
results=json.loads((root/'attribution.json').read_text())
samples=[row for row in results['rows'] if row['kind']=='sampling']
assert len(samples)==8 and all('dropped_bytes differs from 0' in row['quality'] for row in samples)
for row in samples:
    name=f"sampling_s{row['series']}_M{row['M']}_K{row['K']}_N{row['N']}"
    data=json.load(gzip.open(root/'ncu'/(name+'.pc.json.gz'),'rt'))
    sentinel=[pc for pc in data['pcs'] if pc['pc']==2**64-1]
    assert len(sentinel)==1 and sentinel[0]['region']=='unmapped'
    assert sentinel[0]['counts'][analyze.SAMPLE] > 0
print('PASS: real corpus accepted; duplicate PCs, counter corruption, wrong source/phase/template, wrong target, failed launch and missing report rejected')
print('PASS: all eight lossy sampling captures remain inconclusive, and all-ones PC samples stay in the denominator')
