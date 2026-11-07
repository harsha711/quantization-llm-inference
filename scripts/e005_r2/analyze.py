#!/usr/bin/env python3
"""Validate the bounded confirmation and apply its predeclared continuation gate."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

SHAPES=((4096,11008),(11008,4096))
BODY='e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'

def gain(a,c):
    assert math.isfinite(a) and math.isfinite(c) and a>0 and c>0
    return 100*(1-c/a)

def passes(values):
    assert len(values)==2
    return all(v>=5 for v in values)

def checked(v):
    assert v['M']==4 and v['seed']==2 and v['group']==128
    assert v['num_bad']==0 and v['checked']==4*v['N']
    assert v['graph_check'] and v['repeated_launches']==3
    assert all(v[k] for k in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards'))
    attr=v['resources'];grid,shared,capacity={'A':(36,101376,1),'C':(72,49664,2)}[v['config']]
    assert (attr['grid'],attr['dynamic_shared'],attr['max_blocks_per_sm'])==(grid,shared,capacity)
    assert attr['registers']==126 and attr['threads']==256 and attr['static_shared']==0 and attr['local_bytes']==32


def main(r):
    repo=Path(__file__).resolve().parents[2]
    def rows(stem,expected):
        assert (r/(stem+'.exit')).read_text().strip()=='0',stem
        assert BODY in (r/(stem+'.log')).read_text(),stem
        data=[json.loads(x) for x in (r/(stem+'.jsonl')).read_text().splitlines()]
        assert len(data)==len(expected)
        assert {(v['K'],v['N'],v['config'],v['pattern'],v['policy']) for v in data}==expected
        for v in data:checked(v)
        return data
    for stem in ('runtime','identity','identity-final','project-build','project-ctest','project-gpu'):
        assert (r/(stem+'.exit')).read_text().strip()=='0',stem
    assert (r/'dependency-diff.txt').read_text()==''
    assert (r/'project-gpu.log').read_text().count('ERROR SUMMARY: 0 errors')==6
    assert (r/'e006-handoff-check.log').read_text().count(': OK')==113
    assert (r/'e006-preserved-binaries-check.log').read_text().count(': OK')==3
    for stem in ('validation','measurement','final'):
        assert (r/(stem+'-input-check.log')).read_text().count(': OK')==5
    smoke=rows('smoke',{(256,2048,c,p,'no_flush') for c,p in itertools.product('AC',('mixed','zero_a','zero_w'))})
    validation=rows('validation',{(k,n,c,p,'no_flush') for (k,n),c,p in itertools.product(SHAPES,'AC',('mixed','zero_a','zero_w'))})
    sanitizers=[]
    for tool in ('memcheck','racecheck','synccheck'):
        sanitizers+=rows(tool,{(k,n,c,'mixed','no_flush') for (k,n),c in itertools.product(SHAPES,'AC')})
        log=(r/(tool+'.log')).read_text()
        assert 'ERROR SUMMARY: 0 errors' in log
        if tool=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
    expected={(k,n,c,'mixed',p) for (k,n),c,p in itertools.product(SHAPES,'AC',('store_flush','no_flush'))}
    all_rows={};comparison=[];sample_stats=[]
    for series in (1,2):
        data=rows('series'+str(series),expected)
        shapes=SHAPES if series==1 else SHAPES[::-1]
        configs='AC' if series==1 else 'CA'
        policies=('store_flush','no_flush') if series==1 else ('no_flush','store_flush')
        assert [(v['K'],v['N'],v['config'],v['policy']) for v in data]==[(k,n,c,p) for (k,n),c,p in itertools.product(shapes,configs,policies)]
        for v in data:
            assert v['pattern']=='mixed' and v['warmup']==10 and v['iters']==100 and v['method']=='graph_events'
            assert v['order']==configs.index(v['config'])
            samples=v['samples_us'];assert len(samples)==100 and all(math.isfinite(x) and x>0 for x in samples)
            assert statistics.median(samples)==v['median_us'] and sorted(samples)[89]==v['p90_us']
            assert min(samples)==v['min_us'] and max(samples)==v['max_us']
            sample_stats.append(dict(series=series,K=v['K'],N=v['N'],config=v['config'],policy=v['policy'],median_us=v['median_us'],min_us=v['min_us'],max_us=v['max_us'],p90_us=v['p90_us'],stdev_us=float(format(statistics.stdev(samples),'.12g'))))
        all_rows[series]={(v['K'],v['N'],v['config'],v['policy']):v for v in data}
        for k,n in SHAPES:
            group=[v for v in data if (v['K'],v['N'])==(k,n)]
            assert len({v['resources']['kernel_address'] for v in group})==1
            for policy in ('no_flush','store_flush'):
                a=all_rows[series][k,n,'A',policy];c=all_rows[series][k,n,'C',policy]
                assert a['operand_sha256']==c['operand_sha256']
                comparison.append(dict(series=series,K=k,N=n,policy=policy,A_us=a['median_us'],C_us=c['median_us'],gain_pct=gain(a['median_us'],c['median_us']),C_regression_pct=-gain(a['median_us'],c['median_us'])))
    for key,v in all_rows[1].items():
        assert v['operand_sha256']==all_rows[2][key]['operand_sha256']
    gates=[]
    for k,n in SHAPES:
        values=[v['gain_pct'] for v in comparison if (v['K'],v['N'],v['policy'])==(k,n,'no_flush')]
        gates.append(dict(K=k,N=n,no_flush_gain_pct=values,passes=passes(values),minimum_gain_pct=min(values)))
    winners=[v for v in gates if v['passes']]
    winner=max(winners,key=lambda x:x['minimum_gain_pct']) if winners else None
    for filename,values in [('comparisons.csv',comparison),('sample_statistics.csv',sample_stats)]:
        with (r/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
    for line in (r/'measurement-tooling.sha256').read_text().splitlines():
        sha,path=line.split(maxsplit=1)
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha,path
    summary=dict(status='continue with one matched profile pair' if winner else 'stop this lead, neither rectangle passes',
        gate='at least 5% lower C no-flush median in each series, per rectangle',rectangles=gates,
        profile_shape=[winner['K'],winner['N']] if winner else None,body_sha256=BODY,
        timing_rows=16,timed_samples=1600,smoke_rows=len(smoke),fresh_numerical_rows=len(validation),fresh_sanitizer_rows=len(sanitizers),
        max_abs=max(v['max_abs'] for data in all_rows.values() for v in data.values()),
        max_rel_l2=max(v['rel_l2'] for data in all_rows.values() for v in data.values()),
        automatic_dispatch_authorized=False)
    (r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path,nargs='?');p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        assert passes([gain(100,94),gain(100,95)])
        assert not passes([gain(100,94),gain(100,96)])
        assert not passes([gain(100,94),gain(100,106)])
        assert not passes([0,0])
        print('PASS: threshold, missed repeat, order-following and tie checks')
    else:
        assert a.results is not None
        main(a.results)
