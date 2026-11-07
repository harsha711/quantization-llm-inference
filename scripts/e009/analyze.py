#!/usr/bin/env python3
"""Validate fixed E009 evidence and derive fresh matched comparisons."""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('e008_analysis',Path(__file__).resolve().parents[1]/'e008/analyze.py')
e008 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e008)
NAMES = ['marlin','cublas_dq','unfused']
SHAPES = e008.SHAPES


def traversal(shapes, counts, series):
    cells = list(itertools.product(SHAPES,(1,2,4)))
    result = []
    for shape,count in itertools.product(shapes,counts):
        shift = cells.index((shape,count)) % 3 if (shape,count) in cells else (1,2,4).index(count)
        names = NAMES[shift:]+NAMES[:shift]
        for position,name in enumerate(names if series == 1 else names[::-1]):
            result.append((*shape,count,name,position))
    return result


def advantage(marlin, comparator):
    return 100*(1-marlin/comparator)


def checked(v):
    k,n,c = v['K'],v['N'],v['matrices']
    assert (v['M'],v['group'],v['seeds']) == (4,128,[3,4,5,6])
    assert v['body_sha256'] == e008.BODY and v['cublas_math_mode'] == 16
    assert v['method'] == 'complete_cycle_graph_events' and v['policy'] == 'no_artificial_flush'
    assert v['direct_cycles'] == v['poisoned_graph_cycles'] == 3
    assert v['temporary_buffers'] == 1 and v['temporary_bytes'] == 2*k*n
    assert v['exact_individual_gpu_reconstructions'] == 4
    assert v['final_temporary_match'] == (v['config'] == 'unfused')
    for key in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards','temporary_guards'):
        assert v[key]
    metadata = v['allocations']
    assert [a['seed'] for a in metadata] == [3,4,5,6]
    assert len({tuple(a['activation_range']) for a in metadata}) == 1
    assert len({a['operand_sha256']['A'] for a in metadata}) == 1
    ranges = [v['temporary_range'],metadata[0]['activation_range']]
    for a in metadata:
        for key,size in [('packed',k*n//2),('scales',k*n//64),('project_q',k*n//2),
                         ('project_s',k*n//64),('wh',2*k*n),('output',8*n),
                         ('locks',704),('scratch',589952)]:
            # Smoke still has36SM and the same maxM4 workspace capacities.
            interval = a[key+'_range']
            assert interval[1]-interval[0] == size,(key,size,interval)
            ranges.append(interval)
        assert a['project_q_sha256'] == a['operand_sha256']['Q']
        assert a['project_s_sha256'] == a['operand_sha256']['S']
        assert a['wh_sha256'] == a['operand_sha256']['W_hat']
    e008.disjoint(ranges)
    for key in ('packed_sha256','scales_sha256','project_q_sha256','project_s_sha256','wh_sha256'):
        assert len({a[key] for a in metadata}) == 4
    assert v['temporary_range'][1]-v['temporary_range'][0] == 2*k*n+64
    assert metadata[0]['activation_range'][1]-metadata[0]['activation_range'][0] == 8*k
    assert v['live_tensor_storage_bytes'] == sum(b-a for a,b in ranges)
    assert v['framework_reserved_bytes'] >= v['framework_allocated_bytes'] >= v['live_tensor_storage_bytes']
    assert v['active_weight_bytes'] == c*(2*k*n if v['config'] == 'cublas_dq' else k*n//2+k*n//64)
    assert v['l2_bytes'] == 33554432
    assert len(v['resources']) == (c if v['config'] == 'marlin' else 0)
    if k != 256:
        for attr in v['resources']:
            assert (attr['grid'],attr['dynamic_shared'],attr['registers'],attr['threads']) == (36,101376,126,256)
    for key in ('validation','post_timing_validation'):
        if key not in v:
            assert key == 'post_timing_validation' and not v['iters']
            continue
        assert len(v[key]) == c
        for check in v[key]:
            assert check['num_bad'] == 0 and check['checked'] == 4*n
            assert math.isfinite(check['max_abs']) and len(check['output_sha256']) == 64


def main(r,stage):
    def success(name):
        assert (r/(name+'.exit')).read_text().strip() == '0',name
    def rows(name,shapes,counts,series=1,timed=False):
        success(name)
        assert e008.BODY in (r/(name+'.log')).read_text()
        data = [json.loads(x) for x in (r/(name+'.jsonl')).read_text().splitlines()]
        assert [(v['K'],v['N'],v['matrices'],v['config'],v['order']) for v in data] == traversal(shapes,counts,series)
        for v in data:
            checked(v)
            assert v['series'] == series
            assert (v['warmup'],v['iters']) == ((10,100) if timed else (0,0))
            if timed:
                samples = v['samples_us']
                assert len(samples) == 100 and all(math.isfinite(x) and x > 0 for x in samples)
                assert statistics.median(samples) == v['median_us']
                assert (min(samples),max(samples),sorted(samples)[89]) == (v['min_us'],v['max_us'],v['p90_us'])
                assert v['median_per_matrix_us'] == v['median_us']/v['matrices']
                assert [x['output_sha256'] for x in v['validation']] == [x['output_sha256'] for x in v['post_timing_validation']]
            else:
                assert 'samples_us' not in v
        for shape in shapes:
            peers = [v for v in data if (v['K'],v['N']) == shape]
            assert all(v['allocations'] == peers[0]['allocations'] for v in peers)
            assert len({tuple(v['temporary_range']) for v in peers}) == 1
        return data
    success('runtime')
    success('identity')
    assert not (r/'dependency-diff.txt').read_text()
    for name,count in [('e006',113),('e005-r2',148),('e008',120)]:
        assert (r/(name+'-handoff-check.log')).read_text().count(': OK') == count
    identity = json.loads('{'+(r/'identity.log').read_text().split('{',1)[1])
    assert identity['device'] == 'GPU-38a74fc5-9f81-89d1-4ad5-3f5796d39bf2, 610.57.04'
    smoke = rows('smoke',[(256,2048)],[1,2,4])
    if stage == 'prerequisite':
        print('PASS: identity, dependencies, preserved evidence, nine composed smoke rows')
        return
    for name in ('project-build','project-ctest','project-gpu'):
        success(name)
    assert (r/'project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
    validation = rows('validation',SHAPES,[1,2,4])
    for tool in ('memcheck','racecheck','synccheck'):
        rows(tool,SHAPES,[4])
        summary = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert summary in (r/(tool+'.log')).read_text()
    if stage == 'validation':
        print('PASS:18 rectangular numerical rows,18 unfiltered sanitizer rows, retained CUDA checks')
        return
    data = rows('series1',SHAPES,[1,2,4],1,True)+rows('series2',SHAPES[::-1],[4,2,1],2,True)
    success('identity-final')
    for name,count in [('e006',113),('e005-r2',148),('e008',120)]:
        assert (r/('final-'+name+'-check.log')).read_text().count(': OK') == count
    for name in ('validation','measurement','final'):
        assert (r/(name+'-input-check.log')).read_text().count(': OK') == 5
    old = [json.loads(x) for x in (r/'e008-validation-reference.jsonl').read_text().splitlines()]
    for k,n in SHAPES:
        peers = [v for v in data+validation if (v['K'],v['N']) == (k,n)]
        oldmeta = next(v['allocations'] for v in old if (v['K'],v['N']) == (k,n))
        for v in peers:
            for a,b in zip(v['allocations'],oldmeta):
                for key in ('operand_sha256','packed_sha256','scales_sha256'):
                    assert a[key] == b[key]
        for config in NAMES:
            for j in range(4):
                hashes = {v['validation'][j]['output_sha256'] for v in peers if v['config'] == config and v['matrices'] > j}
                assert len(hashes) == 1
    repo = Path(__file__).resolve().parents[2]
    for line in (r/'measurement-tooling.sha256').read_text().splitlines():
        expected,name = line.split(maxsplit=1)
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == expected,name
    comparisons, sample_stats, decisions = [],[],[]
    for v in data:
        sample_stats.append(dict(series=v['series'],K=v['K'],N=v['N'],matrices=v['matrices'],config=v['config'],
            cycle_median_us=v['median_us'],amortized_us=v['median_per_matrix_us'],p90_us=v['p90_us'],
            min_us=v['min_us'],max_us=v['max_us'],stdev_us=statistics.stdev(v['samples_us'])))
    for k,n in SHAPES:
        for count in (1,2,4):
            gains = {name:[] for name in NAMES[1:]}
            for series in (1,2):
                v = {x['config']:x for x in data if (x['K'],x['N'],x['matrices'],x['series']) == (k,n,count,series)}
                row = dict(K=k,N=n,matrices=count,series=series)
                for name in NAMES:
                    row[name+'_cycle_us'] = v[name]['median_us']
                    row[name+'_amortized_us'] = v[name]['median_per_matrix_us']
                for name in NAMES[1:]:
                    row[name+'_marlin_speedup'] = v[name]['median_us']/v['marlin']['median_us']
                    row[name+'_marlin_gain_pct'] = advantage(v['marlin']['median_us'],v[name]['median_us'])
                    gains[name].append(row[name+'_marlin_gain_pct'])
                comparisons.append(row)
            decisions.append(dict(K=k,N=n,matrices=count,gains_pct=gains,
                meaningful={name:e008.meaningful(g) for name,g in gains.items()},
                direction_agrees={name:all(x>0 for x in g) or all(x==0 for x in g) or all(x<0 for x in g) for name,g in gains.items()}))
    for filename,values in [('comparisons.csv',comparisons),('sample_statistics.csv',sample_stats)]:
        with (r/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    summary = dict(timing_rows=36,cycle_samples=3600,timed_gemms=8400,smoke_rows=len(smoke),
        validation_rows=len(validation),sanitizer_rows=18,comparisons=decisions,
        max_abs=max(c['max_abs'] for v in data for c in v['validation']),
        max_rel_l2=max(c['rel_l2'] for v in data for c in v['validation']),
        body_sha256=e008.BODY,next_measurement_authorized=False)
    (r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prerequisite','validation','complete'],default='complete')
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test:
        first = traversal(SHAPES,[1,2,4],1)
        second = traversal(SHAPES[::-1],[4,2,1],2)
        assert [(k,n,c,name) for k,n,c,name,pos in first][::-1] == [(k,n,c,name) for k,n,c,name,pos in second]
        for name in NAMES:
            assert [sum(v[3] == name and v[4] == pos for v in first) for pos in range(3)] == [2,2,2]
        assert e008.meaningful([advantage(95,100),advantage(94,100)])
        assert not e008.meaningful([6,4.9]) and not e008.meaningful([6,-6])
        print('PASS:5% gate, balanced rotations and exactly reversed traversal')
    else:
        assert a.results is not None
        main(a.results,a.stage)
