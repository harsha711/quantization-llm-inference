#!/usr/bin/env python3
"""Gate E013 source, real launch identities, safety evidence and matched timings."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

from prepare_source import ARCHIVE_SHA, NEW, OLD

SHAPES = [(4096,11008),(11008,4096)]


def traversal(shapes, series, patterns=('mixed',)):
    result = []
    for k,n in shapes:
        names = ['A','B'] if (SHAPES.index((k,n)) if (k,n) in SHAPES else 0)%2 == 0 else ['B','A']
        if series == 2:
            names.reverse()
        result.extend((k,n,p,c,i) for p in patterns for i,c in enumerate(names))
    return result


def gain(a, b):
    assert math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0
    return 100*(1-b/a)


def passes(gains):
    return len(gains) == 2 and all(math.isfinite(g) and g >= 5 for g in gains)


def fingerprint(row):
    return [dict(seed=a['seed'],operand_sha256=a['operand_sha256'],
        **{k:a[k] for k in ('packed_sha256','scales_sha256','wh_sha256')}) for a in row['allocations']]


def checked(row, matrices, seeds, timed=False, trace=False):
    k,n,c = row['K'],row['N'],row['config']
    assert c in ('A','B') and (row['M'],row['capacity'],row['group']) == (128,128,128)
    assert row['matrices'] == matrices and row['seeds'] == seeds
    assert row['method'] == 'complete_cycle_graph_events' and row['policy'] == 'no_artificial_flush'
    assert row['direct_cycles'] == row['poisoned_graph_cycles'] == 3
    assert (row['warmup'],row['iters']) == ((10,100) if timed else (0,0))
    assert all(row[key] is True for key in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards'))
    assert row['l2_bytes'] == 33554432
    runtime = row['runtime']
    assert runtime['host'] == 'c4' and runtime['threads'] == 16
    assert runtime['torch'] == '2.14.0+cu130' and runtime['cuda'] == '13.0'
    assert runtime['python'].startswith('3.11.') and runtime['gpu'].startswith('GPU-')
    assert all(re.fullmatch('[0-9a-f]{64}',s) for s in row['identities'].values())
    assert row['identities']['bridge'] == '85f332951a8c24b3eb06594977b3e3d62b7954fa01b986da41ffeff851bfbd3f'
    assert row['identities']['wheel'] == 'e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'
    meta = row['allocations']
    assert len(meta) == matrices and [a['seed'] for a in meta] == seeds
    assert len({tuple(a['activation_range']) for a in meta}) == 1
    assert len({a['operand_sha256']['A'] for a in meta}) == 1
    assert row['activation_sha256'] == meta[0]['operand_sha256']['A']
    ranges = [meta[0]['activation_range']]
    assert ranges[0][1]-ranges[0][0] == 256*k
    for a in meta:
        for key,size in [('packed',k*n//2),('scales',k*n//64),('wh',2*k*n),
                         ('output',256*n),('locks',704),('scratch',36*64*256*4+128)]:
            interval = a[key+'_range']
            assert interval[1]-interval[0] == size
            ranges.append(interval)
        assert a['wh_sha256'] == a['operand_sha256']['W_hat']
    ordered = sorted(ranges)
    assert all(a < b for a,b in ordered)
    assert all(b <= c for (_,b),(c,_) in zip(ordered,ordered[1:]))
    assert row['live_tensor_storage_bytes'] == sum(b-a for a,b in ranges)
    if row['pattern'] == 'mixed':
        for key in ('packed_sha256','scales_sha256','wh_sha256'):
            assert len({a[key] for a in meta}) == matrices
    resources = row['resources']
    assert len(resources) == matrices and all(a == resources[0] for a in resources)
    expected_threads,expected_n = (256,256) if c == 'A' else (128,128)
    for a in resources:
        assert (a['grid'],a['sms'],a['dynamic_shared']) == (36,36,101376)
        assert (a['threads'],a['thread_k'],a['thread_n']) == (expected_threads,64,expected_n)
        assert (a['m_blocks'],a['m8'],a['stages'],a['candidate']) == (4,0,4,0)
        assert a['static_shared'] == 0 and a['max_blocks_per_sm'] == 1
        assert 0 < a['registers'] <= 255 and a['binary_version'] == 120
    if trace:
        assert len(row['traced_kernels']) == matrices and len(set(row['traced_kernels'])) == 1
        symbol = row['traced_kernels'][0]
        assert symbol.startswith('void e005_marlin::Marlin<')
        values = []
        for token in symbol.split('<',1)[1].split('>',1)[0].split(','):
            token = re.sub(r'\([^)]*\)','',token).strip()
            values.append(int(token == 'true') if token in ('true','false') else int(token.rstrip('lLuU')))
        assert values[-8:] == [expected_threads,4,expected_n//16,4,0,4,8,0]
    for key in ('validation','post_timing_validation') if timed else ('validation',):
        assert len(row[key]) == matrices
        for value in row[key]:
            assert value['num_bad'] == 0 and value['checked'] == 128*n
            assert all(math.isfinite(value[x]) for x in ('max_abs','rel_l2','rms'))
            assert re.fullmatch('[0-9a-f]{64}',value['output_sha256'])
    if timed:
        samples = row['samples_us']
        assert len(samples) == 100 and all(math.isfinite(x) and x > 0 for x in samples)
        assert statistics.median(samples) == row['median_us']
        assert (min(samples),max(samples),sorted(samples)[89]) == (row['min_us'],row['max_us'],row['p90_us'])
        assert row['median_per_matrix_us'] == row['median_us']/matrices
        assert [x['output_sha256'] for x in row['validation']] == [x['output_sha256'] for x in row['post_timing_validation']]
    else:
        assert 'samples_us' not in row


def build_checked(root, e006):
    archive = e006/'matched-source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    with tarfile.open(archive) as t:
        original = {m.name.removeprefix('external/patched/'): t.extractfile(m).read()
                    for m in t.getmembers() if m.isfile() and m.name.startswith('external/patched/')}
    with tarfile.open(root/'matched-source.tar.gz') as t:
        actual = {m.name.removeprefix('./'): t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    assert set(actual) == set(original)|{'e013-source.json','e013-host-tile.patch'}
    for name,content in original.items():
        if name == 'csrc/quantization/marlin/marlin.cu':
            changed = actual[name].decode()
            assert changed.count(NEW) == 1 and changed.replace(NEW,OLD).encode() == content
        else:
            assert actual[name] == content,name
    certificate = json.loads(actual['e013-source.json'])
    assert certificate['archive_sha256'] == ARCHIVE_SHA and certificate['device_sources_unchanged']
    for name in original:
        assert certificate['original_sha256'][name] == hashlib.sha256(original[name]).hexdigest()
        assert certificate['generated_sha256'][name] == hashlib.sha256(actual[name]).hexdigest()
    def functions(path):
        return {b.splitlines()[0].strip(): b for b in gzip.open(path,'rt').read().split('Function : ')[1:]}
    saved = functions(e006/'patched/sass.txt.gz')
    current = functions(root/'sass.txt.gz')
    assert len(saved) == 16 and current == saved,'device SASS differs from retained corrected E006'
    digest = (root/'build-binaries.sha256').read_text().split()[0]
    assert hashlib.sha256((root/'libe005.so').read_bytes()).hexdigest() == digest
    return digest


def main(root, e006, stage):
    digest = build_checked(root,e006)
    if stage == 'build':
        print('PASS: one host change, all 16 corrected device functions match E006 SASS')
        return
    def success(name):
        assert (root/(name+'.exit')).read_text().strip() == '0',name
    def rows(name,shapes,matrices=4,seeds=(3,4,5,6),series=1,patterns=('mixed',),timed=False,trace=False):
        success(name)
        data = [json.loads(x) for x in (root/(name+'.jsonl')).read_text().splitlines()]
        assert [(v['K'],v['N'],v['pattern'],v['config'],v['order']) for v in data] == traversal(shapes,series,patterns),name
        for v in data:
            checked(v,matrices,list(seeds),timed,trace)
            assert v['identities']['local'] == digest and v['series'] == series
            if trace:
                path = root/f"{name}_{v['config']}_K{v['K']}_N{v['N']}_{v['pattern']}.trace.json"
                actual = [e['name'] for e in json.loads(path.read_text())['traceEvents'] if e.get('cat') == 'kernel']
                assert actual == v['traced_kernels']
        for shape,pattern in itertools.product(shapes,patterns):
            peers = [v for v in data if (v['K'],v['N'],v['pattern']) == (*shape,pattern)]
            assert all(v['allocations'] == peers[0]['allocations'] for v in peers)
        return data
    success('runtime')
    assert not (root/'dependency-diff.txt').read_text()
    smoke = rows('smoke',[(256,2048)],1,(3,),patterns=('mixed','zero_a','zero_w'),trace=True)
    if stage == 'prerequisite':
        print('PASS: actual A/B launches, six small mixed/zero numerical and graph rows')
        return
    for name in ('project-build','project-ctest','project-gpu'):
        success(name)
    assert (root/'project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
    validation = rows('validation',SHAPES,trace=True)
    safety = []
    for tool in ('memcheck','racecheck','synccheck'):
        data = rows(tool,SHAPES,1 if tool == 'racecheck' else 4,(3,) if tool == 'racecheck' else (3,4,5,6))
        log = (root/(tool+'.log')).read_text()
        summary = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert summary in log and '--error-exitcode 99' in log
        if tool == 'racecheck':
            assert '--kernel-name kns=e005_marlin' in log and '--dump-kernel-launches' not in log
        for v in data:
            peer = next(p for p in validation if (p['K'],p['N'],p['config']) == (v['K'],v['N'],v['config']))
            assert fingerprint(v) == fingerprint(peer)[:v['matrices']]
            assert all({k:x for k,x in a.items() if k != 'kernel_address'} ==
                       {k:x for k,x in peer['resources'][0].items() if k != 'kernel_address'} for a in v['resources'])
        safety.extend(data)
    allrows = smoke+validation+safety
    assert len({json.dumps(v['runtime'],sort_keys=True) for v in allrows}) == 1
    if stage == 'validation':
        print('PASS: four composed numerical, eight composed sanitizer and four representative racecheck rows')
        return
    primary = rows('series1',SHAPES,timed=True)+rows('series2',SHAPES[::-1],series=2,timed=True)
    allrows += primary
    def comparisons(data):
        result = []
        for k,n in SHAPES:
            gains = []
            for series in (1,2):
                pair = {v['config']:v for v in data if (v['K'],v['N'],v['series']) == (k,n,series)}
                gains.append(gain(pair['A']['median_us'],pair['B']['median_us']))
            result.append(dict(K=k,N=n,gains_pct=gains,passes=passes(gains)))
        return result
    primary_decisions = comparisons(primary)
    heldout_decisions = []
    if stage == 'complete' and any(v['passes'] for v in primary_decisions):
        heldout_validation = rows('heldout-validation',SHAPES,seeds=(7,8,9,10),trace=True)
        heldout = rows('heldout1',SHAPES,seeds=(7,8,9,10),timed=True)+rows(
            'heldout2',SHAPES[::-1],seeds=(7,8,9,10),series=2,timed=True)
        allrows += heldout_validation+heldout
        heldout_decisions = comparisons(heldout)
    elif stage == 'complete':
        assert not any((root/(name+'.jsonl')).exists() for name in ('heldout-validation','heldout1','heldout2'))
    for k,n in SHAPES:
        peers = [v for v in allrows if (v['K'],v['N']) == (k,n)]
        for seeds in ([3],[3,4,5,6],[7,8,9,10]):
            group = [v for v in peers if v['seeds'] == seeds]
            if group:
                assert all(fingerprint(v) == fingerprint(group[0]) for v in group)
    assert len({json.dumps(v['runtime'],sort_keys=True) for v in allrows}) == 1
    for filename in ('measurement-inputs.sha256','measurement-tooling.sha256'):
        for line in (root/filename).read_text().splitlines():
            expected,name = line.split(maxsplit=1)
            path = (root if filename == 'measurement-inputs.sha256' else
                    Path(__file__).resolve().parents[2])/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected,name
    summary = dict(stage=stage,primary=primary_decisions,heldout=heldout_decisions,
        heldout_required=any(v['passes'] for v in primary_decisions),
        default='corrected E006 A', automatic_dispatch=False,
        timed_rows=sum(v['iters'] > 0 for v in allrows),
        restricted_improvements=[dict(K=p['K'],N=p['N']) for p,h in zip(primary_decisions,heldout_decisions)
                                 if p['passes'] and h['passes']])
    (root/('primary-analysis.json' if stage == 'primary' else 'analysis.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--e006',type=Path,default=Path(__file__).resolve().parents[2]/'results/e006_20261106')
    p.add_argument('--stage',choices=['build','prerequisite','validation','primary','complete'],default='complete')
    a = p.parse_args()
    main(a.results,a.e006,a.stage)
