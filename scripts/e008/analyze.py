#!/usr/bin/env python3
"""Check complete E008 evidence and apply the predeclared per-pattern gate."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
SHAPES = [(4096,11008),(11008,4096)]


def disjoint(ranges):
    ordered = sorted(ranges)
    assert all(a < b for a, b in ordered)
    assert all(b <= c for (_, b), (c, _) in zip(ordered, ordered[1:]))


def meaningful(gains):
    assert len(gains) == 2
    return all(g >= 5 for g in gains)


def checked(v):
    assert (v['M'],v['group'],v['seeds']) == (4,128,[3,4,5,6])
    assert v['body_sha256'] == BODY
    assert v['method'] == 'complete_cycle_graph_events' and v['policy'] == 'no_artificial_flush'
    count = v['matrices']
    assert count in (1,2,4) and v['direct_cycles'] == v['poisoned_graph_cycles'] == 3
    assert all(v[k] for k in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards'))
    metadata = v['allocations']
    assert len(metadata) == 4 and [a['seed'] for a in metadata] == [3,4,5,6]
    assert len({a['operand_sha256']['A'] for a in metadata}) == 1
    assert len({tuple(a['activation_range']) for a in metadata}) == 1
    disjoint([a[key] for a in metadata for key in ('packed_range','scales_range','output_range','locks_range','scratch_range')]+[metadata[0]['activation_range']])
    for key in ('packed_sha256','scales_sha256'):
        assert len({a[key] for a in metadata}) == 4
    for a in metadata:
        assert a['packed_bytes'] == v['K']*v['N']//2
        assert a['scales_bytes'] == 2*(v['K']//128)*v['N']
        for key in ('packed','scales'):
            assert a[key+'_range'][1]-a[key+'_range'][0] == a[key+'_bytes']
    assert v['active_weight_bytes'] == count*(v['K']*v['N']//2+2*(v['K']//128)*v['N'])
    assert v['l2_bytes'] == 33554432
    assert len(v['resources']) == len(v['validation']) == count
    for attr in v['resources']:
        grid,shared,capacity = {'A':(36,101376,1),'C':(72,49664,2)}[v['config']]
        assert (attr['grid'],attr['dynamic_shared'],attr['max_blocks_per_sm']) == (grid,shared,capacity)
        assert (attr['registers'],attr['threads'],attr['static_shared'],attr['local_bytes']) == (126,256,0,32)
    for checks in (v['validation'],v.get('post_timing_validation',[])):
        for check in checks:
            assert check['num_bad'] == 0 and check['checked'] == 4*v['N']
            assert len(check['output_sha256']) == 64


def main(r, stage):
    def success(name):
        assert (r/(name+'.exit')).read_text().strip() == '0', name
    def rows(name, shapes, counts, configs, timed=False):
        success(name)
        assert BODY in (r/(name+'.log')).read_text()
        data = [json.loads(x) for x in (r/(name+'.jsonl')).read_text().splitlines()]
        assert [(v['K'],v['N'],v['matrices'],v['config']) for v in data] == [(k,n,c,b) for (k,n),c,b in itertools.product(shapes,counts,configs)]
        for v in data:
            checked(v)
            assert v['order'] == configs.index(v['config'])
            assert (v['warmup'],v['iters']) == ((10,100) if timed else (0,0))
            if timed:
                samples = v['samples_us']
                assert len(samples) == 100 and all(math.isfinite(x) and x > 0 for x in samples)
                assert statistics.median(samples) == v['median_us']
                assert (min(samples),max(samples),sorted(samples)[89]) == (v['min_us'],v['max_us'],v['p90_us'])
                assert v['median_per_matrix_us'] == v['median_us']/v['matrices']
                assert len(v['post_timing_validation']) == v['matrices']
                assert [x['output_sha256'] for x in v['validation']] == [x['output_sha256'] for x in v['post_timing_validation']]
            else:
                assert 'samples_us' not in v
        for k,n in shapes:
            group = [v for v in data if (v['K'],v['N']) == (k,n)]
            assert all(v['allocations'] == group[0]['allocations'] for v in group)
            assert len({a['kernel_address'] for v in group for a in v['resources']}) == 1
        return data
    success('runtime')
    success('identity')
    assert not (r/'dependency-diff.txt').read_text()
    assert (r/'e006-handoff-check.log').read_text().count(': OK') == 113
    assert (r/'e005-r2-handoff-check.log').read_text().count(': OK') == 148
    identity = json.loads('{'+(r/'identity.log').read_text().split('{',1)[1])
    assert 'GPU-38a74fc5-9f81-89d1-4ad5-3f5796d39bf2, 610.57.04' == identity['device']
    smoke = rows('smoke',[(256,2048)],[1,2,4],'AC')
    if stage == 'prerequisite':
        print('PASS: identity, exact dependencies and six complete-cycle smoke rows')
        return
    for name in ('project-build','project-ctest','project-gpu'):
        success(name)
    assert (r/'project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors') == 6
    validation = rows('validation',SHAPES,[1,2,4],'AC')
    for tool in ('memcheck','racecheck','synccheck'):
        rows(tool,SHAPES,[4],'AC')
        summary = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
        assert summary in (r/(tool+'.log')).read_text()
    if stage == 'validation':
        print('PASS: 12 rectangular numerical rows, 12 sanitizer rows and retained CUDA checks')
        return
    success('identity-final')
    for name in ('validation','measurement','final'):
        assert (r/(name+'-input-check.log')).read_text().count(': OK') == 5
    data = rows('series1',SHAPES,[1,2,4],'AC',True)
    second = rows('series2',SHAPES[::-1],[4,2,1],'CA',True)
    comparisons = []
    sample_stats = []
    for series, values in enumerate((data,second),1):
        for v in values:
            sample_stats.append(dict(series=series,K=v['K'],N=v['N'],matrices=v['matrices'],config=v['config'],
                cycle_median_us=v['median_us'],per_matrix_us=v['median_per_matrix_us'],
                min_us=v['min_us'],max_us=v['max_us'],p90_us=v['p90_us'],stdev_us=statistics.stdev(v['samples_us'])))
        for k,n in SHAPES:
            for count in (1,2,4):
                a,c = [next(v for v in values if (v['K'],v['N'],v['matrices'],v['config']) == (k,n,count,b)) for b in 'AC']
                assert a['allocations'] == c['allocations']
                comparisons.append(dict(series=series,K=k,N=n,matrices=count,A_cycle_us=a['median_us'],C_cycle_us=c['median_us'],
                    A_per_matrix_us=a['median_per_matrix_us'],C_per_matrix_us=c['median_per_matrix_us'],gain_pct=100*(1-c['median_us']/a['median_us'])))
    for k,n in SHAPES:
        peers = [v for v in data+second+validation if (v['K'],v['N']) == (k,n)]
        for v in peers:
            for a,b in zip(v['allocations'],peers[0]['allocations']):
                for key in ('operand_sha256','packed_sha256','scales_sha256'):
                    assert a[key] == b[key]
        for config in 'AC':
            peers_config = [v for v in peers if v['config'] == config]
            for j in range(4):
                outputs = [v['validation'][j]['output_sha256'] for v in peers_config if v['matrices'] > j]
                assert len(set(outputs)) == 1
    decisions = []
    for k,n in SHAPES:
        gains = {count:[v['gain_pct'] for v in comparisons if (v['K'],v['N'],v['matrices']) == (k,n,count)] for count in (1,2,4)}
        passed = {c:meaningful(g) for c,g in gains.items()}
        losses = [c for c in (2,4) if all(g < 0 for g in gains[c])]
        if not passed[1]:
            decision = 'single-matrix control did not reproduce meaningful gain'
        elif losses:
            decision = 'reuse-dependent gain, retain A and close lead for reproducibly losing cycle lengths'
        elif all(passed.values()):
            decision = 'meaningful gain survives all tested cycle lengths, restricted policy investigation may be proposed'
        elif not passed[2] and not passed[4]:
            decision = 'only single-matrix meaningful gain confirmed, no general decoding claim'
        else:
            decision = 'mixed cycle-length evidence, no general policy conclusion'
        decisions.append(dict(K=k,N=n,gains_pct=gains,passes=passed,reproducibly_losing_counts=losses,decision=decision))
    repo = Path(__file__).resolve().parents[2]
    for line in (r/'measurement-tooling.sha256').read_text().splitlines():
        sha,name = line.split(maxsplit=1)
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == sha, name
    for filename, values in [('comparisons.csv',comparisons),('sample_statistics.csv',sample_stats)]:
        with (r/filename).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    summary = dict(timing_rows=24,cycle_samples=2400,timed_gemms=5600,body_sha256=BODY,
        smoke_rows=len(smoke),numerical_rows=len(validation),sanitizer_rows=12,rectangles=decisions,
        automatic_dispatch_authorized=False,max_abs=max(c['max_abs'] for v in data+second for c in v['validation']))
    (r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path,nargs='?')
    p.add_argument('--stage',choices=['prerequisite','validation','complete'],default='complete')
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test:
        assert meaningful([5,6]) and not meaningful([5,4.9]) and not meaningful([6,-6])
        disjoint([[10,20],[20,30]])
        try:
            disjoint([[10,20],[10,20]])
        except AssertionError:
            pass
        else:
            raise AssertionError('alias accepted')
        print('PASS: fixed gate and storage alias rejection')
    else:
        assert a.results is not None
        main(a.results,a.stage)
