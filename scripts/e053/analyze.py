#!/usr/bin/env python3
"""Check complete E053 samples and parent bits, then apply the fixed 12-cell gate."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import statistics

SHAPES = [(4096,4096),(4096,11008),(11008,4096)]
MS = [1,2,4,8,16]
NAMES = ['v3','v3_m2','tc_ladder','marlin','cublas_dq']


def check_row(r):
    assert r['matrices'] == 4 and r['group'] == 128 and r['seeds'] == [3,4,5,6]
    assert r['method'] == 'complete_cycle_graph_events' and r['policy'] == 'no_artificial_flush'
    assert r['warmup'] == 10 and r['iters'] == 100 and r['cublas_math_mode'] == 16
    assert r['output_tail_untouched'] and r['workspace_guards'] and r['lock_reset']
    assert r['direct_cycles'] == r['poisoned_graph_cycles'] == 3
    assert len(r['validation']) == 4 and r['post_timing_validation'] == r['validation']
    for v in r['validation']:
        assert v['num_bad'] == 0 and v['checked'] == r['M']*r['N']
        assert math.isfinite(v['rel_l2']) and len(v['output_sha256']) == 64
    samples = r['samples_us']
    assert len(samples) == 100 and all(math.isfinite(x) and x > 0 for x in samples)
    assert r['median_us'] == statistics.median(samples)
    assert r['min_us'] == min(samples) and r['max_us'] == max(samples)
    assert r['p90_us'] == sorted(samples)[89]
    assert r['median_per_matrix_us'] == r['median_us']/4
    shift = (SHAPES.index((r['K'],r['N'])) + MS.index(r['M'])) % len(NAMES)
    order = NAMES[shift:] + NAMES[:shift]
    if r['series'] == 2:
        order.reverse()
    assert order[r['order']] == r['config']


def analyze(root):
    by = {}
    identity = {}
    libraries = set()
    for series in (1,2):
        rows = [json.loads(x) for x in (root/f'series{series}.jsonl').read_text().splitlines()]
        expected = {(k,n,m,c) for k,n in SHAPES for m in MS for c in NAMES}
        assert len(rows) == len(expected) == 75
        assert {(r['K'],r['N'],r['M'],r['config']) for r in rows} == expected
        for r in rows:
            assert r['series'] == series
            check_row(r)
            k,n,m,c = r['K'],r['N'],r['M'],r['config']
            by[(series,k,n,m,c)] = r
            operand = ([a['operand_sha256'] for a in r['allocations']],r['activation_prefix_sha256'])
            assert identity.setdefault((k,n,m),operand) == operand
            libraries.add(r['kernels_sha256'])
    assert len(libraries) == 1
    comparisons = []
    for series in (1,2):
        for k,n in SHAPES:
            for m in MS:
                parent = by[series,k,n,m,'v3']
                candidate = by[series,k,n,m,'v3_m2']
                assert [v['output_sha256'] for v in candidate['validation']] == [v['output_sha256'] for v in parent['validation']]
                gain = 100*(1-candidate['median_us']/parent['median_us'])
                row = dict(series=series,K=k,N=n,M=m,v3_us=parent['median_us']/4,
                           v3_m2_us=candidate['median_us']/4,gain_pct=gain,primary=m in (4,8))
                for name in ('tc_ladder','marlin','cublas_dq'):
                    row[name+'_us'] = by[series,k,n,m,name]['median_us']/4
                comparisons.append(row)
    primary = [v for v in comparisons if v['primary']]
    assert len(primary) == 12
    passed = sum(v['v3_m2_us'] <= .95*v['v3_us'] for v in primary)
    result = dict(decision='ACCEPTED' if passed == 12 else 'REJECTED',primary_passes=passed,
                  primary_cells=12,timing_rows=150,cycle_samples=15000,parent_bitwise_cells=30,
                  kernel_library_sha256=next(iter(libraries)),comparisons=comparisons)
    return result


def self_test():
    row = dict(matrices=4,group=128,seeds=[3,4,5,6],method='complete_cycle_graph_events',
               policy='no_artificial_flush',warmup=10,iters=100,cublas_math_mode=16,
               output_tail_untouched=True,workspace_guards=True,lock_reset=True,
               direct_cycles=3,poisoned_graph_cycles=3,M=4,N=4096,K=4096,series=1,config='tc_ladder',order=0,
               validation=[dict(num_bad=0,checked=4*4096,rel_l2=0,output_sha256='0'*64)]*4,
               samples_us=[4.0]*100,median_us=4.0,min_us=4.0,max_us=4.0,p90_us=4.0,median_per_matrix_us=1.0)
    row['post_timing_validation'] = copy.deepcopy(row['validation'])
    check_row(row)
    mutations = [('samples_us',[4.0]*99),('samples_us',[float('nan')]*100),('median_us',5.0),
                 ('cublas_math_mode',0),('output_tail_untouched',False)]
    for key,value in mutations:
        bad = copy.deepcopy(row); bad[key] = value
        try:
            check_row(bad)
        except AssertionError:
            continue
        raise AssertionError(f'accepted corrupt {key}')
    assert 95 <= .95*100 and not 95.0001 <= .95*100
    print('PASS: valid row accepted; missing/nonfinite samples, changed median, math mode and guard rejected')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path,nargs='?')
    p.add_argument('--self-test',action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.root
        result = analyze(args.root)
        (args.root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
        with (args.root/'comparisons.csv').open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(result['comparisons'][0]),lineterminator='\n')
            w.writeheader(); w.writerows(result['comparisons'])
        print(json.dumps({k:v for k,v in result.items() if k != 'comparisons'},indent=2))
