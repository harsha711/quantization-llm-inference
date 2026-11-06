#!/usr/bin/env python3
"""Assert E004 series completeness and regenerate compact comparisons from samples."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SHAPES=[(4096,4096),(4096,11008),(11008,4096)]
MS=[1,2,4,8,16,32,64,128]
KERNELS=['marlin','naive','inblock_specialized','inblock_m2','inblock_m4','cublas','cublas_dq','unfused']
POLICIES=['store_flush','no_flush']


def key(r):return r['K'],r['N'],r['M'],r['policy'],r['kernel']


def load(path):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    assert len({key(r) for r in rows})==len(rows),f'duplicate rows in {path}'
    for r in rows:
        assert r['pass_check'] and r['num_bad']==0 and r['checked']==r['M']*r['N']
        assert r['repack_codes_match'] and r['repack_scale_bits_match'] and r['full_control_match']
        assert r['reconstructed_elements']==r['K']*r['N']
        assert r['group']==128 and r['seed']==1 and r['cublas_math_mode']==16
        assert r['warmup']==10
        s=r['samples_us'];assert len(s)==r['iters'] and all(math.isfinite(x) and x>0 for x in s)
        assert statistics.median(s)==r['median_us']
        assert sorted(s)[math.floor(.9*(len(s)-1))]==r['p90_us']
        assert min(s)==r['min_us'] and max(s)==r['max_us']
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(root):
    rows=load(root/'screen.jsonl')
    expected={(k,n,m,p,c) for k,n in SHAPES for m in MS for p in POLICIES for c in KERNELS}
    assert {key(r) for r in rows}==expected and len(rows)==384
    assert all(r['iters']==20 and r['method']=='graph_events' for r in rows)
    lookup={key(r):r for r in rows}
    table=[]
    for k,n in SHAPES:
        for m in MS:
            row={'K':k,'N':n,'M':m}
            for policy in POLICIES:
                d={c:lookup[k,n,m,policy,c]['median_us'] for c in KERNELS}
                fused=min(['naive','inblock_specialized','inblock_m2','inblock_m4'],key=d.get)
                row.update({f'{policy}_marlin_us':d['marlin'],f'{policy}_cublas_us':d['cublas'],
                            f'{policy}_cublas_dq_us':d['cublas_dq'],f'{policy}_best_retained':fused,
                            f'{policy}_best_retained_us':d[fused],f'{policy}_retained_over_marlin':d[fused]/d['marlin'],
                            f'{policy}_cublas_dq_over_marlin':d['cublas_dq']/d['marlin']})
            table.append(row)
    write_csv(root/'performance_map.csv',table)
    # Each logical weight set is identical across all comparators/M/policies.
    for k,n in SHAPES:
        signatures={json.dumps(r['operand_sha256'],sort_keys=True) for r in rows if (r['K'],r['N'])==(k,n)}
        assert len(signatures)==1
    summary=dict(screen_rows=len(rows),screen_complete=True,all_outputs_checked=True,
                 max_relative_l2=max(r['rel_l2'] for r in rows),max_absolute_error=max(r['max_abs'] for r in rows))
    # Repeated target sets are checked after the diagnostic selection is recorded.
    for name in ['repeat1','repeat2','eager_diagnostic','graph_diagnostic']:
        path=root/(name+'.jsonl')
        if not path.exists():continue
        other=load(path)
        assert all(r['iters']==100 for r in other)
        for r in other:
            assert key(r) in lookup
            base=lookup[key(r)]
            for operand in ['W','Q','S','W_hat']:
                assert base['operand_sha256'][operand]==r['operand_sha256'][operand]
        summary[name+'_rows']=len(other)
    if (root/'repeat1.jsonl').exists() and (root/'repeat2.jsonl').exists():
        a={key(r):r for r in load(root/'repeat1.jsonl')};b={key(r):r for r in load(root/'repeat2.jsonl')}
        assert a.keys()==b.keys()
        comps=[]
        for x in a:
            r,s=a[x],b[x]
            assert r['method']==s['method']=='graph_events'
            assert r['operand_sha256']==s['operand_sha256']
            comps.append(dict(K=x[0],N=x[1],M=x[2],policy=x[3],kernel=x[4],screen_us=lookup[x]['median_us'],repeat1_us=r['median_us'],repeat2_us=s['median_us'],repeat2_over_repeat1=s['median_us']/r['median_us']))
        write_csv(root/'repeat_comparison.csv',comps)
    (root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args();main(a.results)
