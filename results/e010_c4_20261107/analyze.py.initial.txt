#!/usr/bin/env python3
"""Check E010 paired evidence, actual selections and bounded crossover refinement."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics

spec = importlib.util.spec_from_file_location('e008_analysis',Path(__file__).resolve().parents[1]/'e008/analyze.py')
e008 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e008)
SHAPES = [(4096,11008),(11008,4096)]
MS = [4,16,64,128]
NAMES = ['marlin','cublas_dq']


def traversal(shapes,ms,series,names=NAMES):
    result = []
    for (k,n),m in itertools.product(shapes,ms):
        parity = ((SHAPES.index((k,n)) if (k,n) in SHAPES else 0)+(MS.index(m) if m in MS else 0))%2
        order = NAMES[parity:]+NAMES[:parity]
        for pos,name in enumerate(order if series == 1 else order[::-1]):
            if name in names:
                result.append((k,n,m,name,pos))
    return result


def direction(gains):
    return 'marlin' if all(g>0 for g in gains) else 'cublas_dq' if all(g<0 for g in gains) else 'inconclusive'


def midpoints(decisions):
    result = []
    for k,n in SHAPES:
        by_m = {v['M']:v['winner'] for v in decisions if (v['K'],v['N']) == (k,n)}
        for a,b in zip(MS,MS[1:]):
            if by_m[a] != 'inconclusive' and by_m[b] != 'inconclusive' and by_m[a] != by_m[b]:
                result.append(dict(K=k,N=n,lower=a,upper=b,midpoint=(a+b)//2))
    return result


def checked(v):
    k,n,m = v['K'],v['N'],v['M']
    assert (v['capacity'],v['matrices'],v['group'],v['seeds']) == (128,4,128,[3,4,5,6])
    assert v['body_sha256'] == e008.BODY and v['cublas_math_mode'] == 16
    assert (v['method'],v['policy']) == ('complete_cycle_graph_events','no_artificial_flush')
    assert v['direct_cycles'] == v['poisoned_graph_cycles'] == 3
    for key in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards','output_tail_untouched'):
        assert v[key]
    metadata = v['allocations']
    assert [a['seed'] for a in metadata] == [3,4,5,6]
    assert len({tuple(a['activation_range']) for a in metadata}) == 1
    assert len({a['operand_sha256']['A'] for a in metadata}) == 1
    ranges = [metadata[0]['activation_range']]
    assert ranges[0][1]-ranges[0][0] == 256*k
    for a in metadata:
        for key,size in [('packed',k*n//2),('scales',k*n//64),('wh',2*k*n),
                         ('output',256*n),('locks',704),('scratch',36*64*256*4+128)]:
            interval = a[key+'_range']
            assert interval[1]-interval[0] == size,(key,size,interval)
            ranges.append(interval)
        assert a['wh_sha256'] == a['operand_sha256']['W_hat']
    e008.disjoint(ranges)
    for key in ('packed_sha256','scales_sha256','wh_sha256'):
        assert len({a[key] for a in metadata}) == 4
    assert v['live_tensor_storage_bytes'] == sum(b-a for a,b in ranges)
    assert v['framework_reserved_bytes'] >= v['framework_allocated_bytes'] >= v['live_tensor_storage_bytes']
    assert v['active_weight_bytes'] == 4*(2*k*n if v['config'] == 'cublas_dq' else k*n//2+k*n//64)
    assert v['l2_bytes'] == 33554432
    assert len(v['resources']) == (4 if v['config'] == 'marlin' else 0)
    assert all(a == v['resources'][0] for a in v['resources'])
    for a in v['resources']:
        assert a['candidate'] == 0 and a['sms'] == 36 and a['stages'] == 4
        assert a['m8'] == int(m<=8)
        assert a['m_blocks'] == min((m+15)//16,4)
    if v['traced_kernels']:
        assert v['config'] == 'marlin' and len(v['traced_kernels']) == 1
        symbol = v['traced_kernels'][0]
        assert symbol.startswith('void e005_marlin::Marlin<')
        values = [int(re.sub(r'\([^)]*\)|[lLuU]','',x).strip()) for x in symbol.split('<',1)[1].split('>',1)[0].split(',')]
        a = v['resources'][0]
        assert values[-8:] == [a['threads'],a['m_blocks'],a['thread_n']//16,a['thread_k']//16,a['m8'],a['stages'],8,0],(values,a)
    for key in ('validation','post_timing_validation'):
        if key not in v:
            assert key == 'post_timing_validation' and v['iters'] == 0
            continue
        assert len(v[key]) == 4
        for c in v[key]:
            assert c['num_bad'] == 0 and c['checked'] == m*n
            assert math.isfinite(c['max_abs']) and len(c['output_sha256']) == 64


def main(r,stage):
    def success(name):
        assert (r/(name+'.exit')).read_text().strip() == '0',name
    def rows(name,shapes,ms,series=1,timed=False,names=NAMES):
        success(name)
        data = [json.loads(x) for x in (r/(name+'.jsonl')).read_text().splitlines()]
        assert [(v['K'],v['N'],v['M'],v['config'],v['order']) for v in data] == traversal(shapes,ms,series,names),name
        for v in data:
            checked(v)
            assert v['series'] == series
            assert (v['warmup'],v['iters']) == ((10,100) if timed else (0,0))
            if timed:
                s = v['samples_us']
                assert len(s) == 100 and all(math.isfinite(x) and x>0 for x in s)
                assert statistics.median(s) == v['median_us']
                assert (min(s),max(s),sorted(s)[89]) == (v['min_us'],v['max_us'],v['p90_us'])
                assert v['median_per_matrix_us'] == v['median_us']/4
                assert [x['output_sha256'] for x in v['validation']] == [x['output_sha256'] for x in v['post_timing_validation']]
            else:
                assert 'samples_us' not in v
        for shape in shapes:
            peers = [v for v in data if (v['K'],v['N']) == shape]
            assert all(v['allocations'] == peers[0]['allocations'] for v in peers)
        return data
    for name in ('runtime','identity','library-identity','kernel-symbols'):
        success(name)
    assert not (r/'dependency-diff.txt').read_text()
    for name,count in [('e006',113),('e005-r2',148),('e008',120),('e009',206)]:
        assert (r/(name+'-handoff-check.log')).read_text().count(': OK') == count
    smoke = rows('smoke',[(256,2048)],MS)
    if stage == 'prerequisite':
        print('PASS: preserved identities, exact dependencies, eight small full-cycle rows')
        return
    for name in ('project-build','project-ctest','project-gpu'):
        success(name)
    assert (r/'project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
    validation = rows('validation',SHAPES,MS)
    for v in validation:
        assert len(v['traced_kernels']) == (1 if v['config'] == 'marlin' else 0)
    for tool in ('memcheck','racecheck','synccheck'):
        rows(tool,SHAPES,MS,names=['marlin'])
        text = (r/(tool+'.log')).read_text()
        assert ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors') in text
        if tool == 'racecheck':
            assert '--kernel-name kns=e005_marlin' in text and '--dump-kernel-launches' not in text
    if stage == 'validation':
        print('PASS:16 numerical /24 custom-kernel sanitizer rows, eight traced actual specializations, retained CUDA checks')
        return
    data = rows('series1',SHAPES,MS,1,True)+rows('series2',SHAPES[::-1],MS[::-1],2,True)
    for label in ('measurement','final'):
        assert (r/(label+'-library-check.log')).read_text().count(': OK') >= 6
    for label in ('validation','measurement','final'):
        assert (r/(label+'-input-check.log')).read_text().count(': OK') == 5
    success('identity-final')
    repo = Path(__file__).resolve().parents[2]
    for line in (r/'measurement-tooling.sha256').read_text().splitlines():
        digest,name = line.split(maxsplit=1)
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest,name
    for k,n in SHAPES:
        peers = [v for v in data+validation if (v['K'],v['N']) == (k,n)]
        for v in peers:
            for a,b in zip(v['allocations'],peers[0]['allocations']):
                for key in ('operand_sha256','packed_sha256','scales_sha256','wh_sha256'):
                    assert a[key] == b[key]
        for m,name in itertools.product(MS,NAMES):
            values = [v for v in peers if (v['M'],v['config']) == (m,name)]
            assert len({v['activation_prefix_sha256'] for v in values}) == 1
            assert len({tuple(c['output_sha256'] for c in v['validation']) for v in values}) == 1
            if name == 'marlin':
                attrs = [{key:value for key,value in v['resources'][0].items() if key != 'kernel_address'} for v in values]
                assert all(a == attrs[0] for a in attrs)
    comparisons,decisions = [],[]
    for k,n in SHAPES:
        for m in MS:
            gains = []
            for series in (1,2):
                v = {x['config']:x for x in data if (x['K'],x['N'],x['M'],x['series']) == (k,n,m,series)}
                gain = 100*(1-v['marlin']['median_us']/v['cublas_dq']['median_us'])
                gains.append(gain)
                comparisons.append(dict(K=k,N=n,M=m,series=series,
                    marlin_cycle_us=v['marlin']['median_us'],cublas_cycle_us=v['cublas_dq']['median_us'],
                    marlin_amortized_us=v['marlin']['median_us']/4,cublas_amortized_us=v['cublas_dq']['median_us']/4,
                    speedup=v['cublas_dq']['median_us']/v['marlin']['median_us'],marlin_gain_pct=gain))
            decisions.append(dict(K=k,N=n,M=m,gains_pct=gains,winner=direction(gains),meaningful_marlin=all(g>=5 for g in gains)))
    sample_stats = [dict(K=v['K'],N=v['N'],M=v['M'],series=v['series'],config=v['config'],
        median_us=v['median_us'],p90_us=v['p90_us'],min_us=v['min_us'],max_us=v['max_us'],
        stdev_us=statistics.stdev(v['samples_us'])) for v in data]
    for name,values in [('comparisons.csv',comparisons),('sample_statistics.csv',sample_stats)]:
        with (r/name).open('w',newline='') as out:
            writer = csv.DictWriter(out,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader(); writer.writerows(values)
    summary = dict(timing_rows=len(data),cycle_samples=len(data)*100,timed_gemms=len(data)*400,
        decisions=decisions,crossover_brackets=midpoints(decisions),
        max_abs=max(c['max_abs'] for v in data for c in v['validation']),
        max_rel_l2=max(c['rel_l2'] for v in data for c in v['validation']))
    (r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prerequisite','validation','complete'],default='complete')
    p.add_argument('--self-test',action='store_true')
    args = p.parse_args()
    if args.self_test:
        a = traversal(SHAPES,MS,1)
        b = traversal(SHAPES[::-1],MS[::-1],2)
        assert [x[:4] for x in a][::-1] == [x[:4] for x in b]
        for name in NAMES:
            assert [sum(x[3] == name and x[4] == pos for x in a) for pos in (0,1)] == [4,4]
        assert direction([1,2]) == 'marlin' and direction([-1,-2]) == 'cublas_dq'
        assert direction([1,-2]) == direction([0,0]) == 'inconclusive'
        fake = [dict(K=k,N=n,M=m,winner='marlin' if m<=16 else 'cublas_dq') for k,n in SHAPES for m in MS]
        assert [x['midpoint'] for x in midpoints(fake)] == [40,40]
        assert not midpoints([dict(v,winner='marlin') for v in fake])
        print('PASS: balanced reversed traversal, agreed-direction crossover and bounded midpoint rule')
    else:
        assert args.results is not None
        main(args.results,args.stage)
