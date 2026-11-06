#!/usr/bin/env python3
"""Check complete E007 matched-body evidence and apply its predeclared criterion."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
p.add_argument('--e006',type=Path,default=Path(__file__).resolve().parents[2]/'results/e006_20261106')
a=p.parse_args();r=a.root
repo=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(Path(__file__).with_name('check.py')),str(r),'--stage','validation','--e006',str(a.e006)],check=True)
build=json.loads((r/'build-check.json').read_text());hashes=build['binary_sha256']
policies=('store_flush','no_flush');shapes=((4096,4096),(4096,11008),(11008,4096))

def write_csv(name,rows):
    with (r/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

data={};table=[];stats=[]
for stem,ms,dims,seed in [('series1',(1,8),shapes[:1],1),('series2',(1,8),shapes[:1],1),('fallback1',(16,64),shapes[:1],1),('fallback2',(16,64),shapes[:1],1),('heldout',(2,4),shapes,2)]:
    paired={}
    for variant in ('control','candidate'):
        label=variant+'_'+stem
        assert (r/(label+'.exit')).read_text().strip()=='0',label
        rows=[json.loads(x) for x in (r/(label+'.jsonl')).read_text().splitlines()]
        expected={(m,k,n,p) for m,(k,n),p in itertools.product(ms,dims,policies)}
        assert len(rows)==len(expected) and {(v['M'],v['K'],v['N'],v['policy']) for v in rows}==expected
        assert hashes[variant] in (r/(label+'.log')).read_text()
        for v in rows:
            assert v['config']=='A' and v['seed']==seed and v['group']==128 and v['pattern']=='mixed'
            assert v['num_bad']==0 and v['checked']==v['M']*v['N'] and v['graph_check'] and v['repeated_launches']==3
            assert v['warmup']==10 and v['iters']==100 and v['method']=='graph_events'
            assert all(v[k] for k in ('codes_match','scales_match','reconstruction_match','lock_reset','workspace_guards'))
            samples=v['samples_us'];assert len(samples)==100 and all(math.isfinite(x) and x>0 for x in samples)
            assert statistics.median(samples)==v['median_us'] and sorted(samples)[89]==v['p90_us']
            assert min(samples)==v['min_us'] and max(samples)==v['max_us']
            attr=v['resources'];assert attr['grid']==36 and attr['dynamic_shared']==101376 and attr['threads']==256
            assert attr['static_shared']==0
            if v['M']<=8:assert attr['registers']==(126 if variant=='control' else 124) and attr['local_bytes']==32
            stats.append(dict(series=stem,variant=variant,M=v['M'],K=v['K'],N=v['N'],policy=v['policy'],
                median_us=v['median_us'],min_us=v['min_us'],max_us=v['max_us'],p90_us=v['p90_us'],
                sample_variance_us2=float(format(statistics.variance(samples),'.12g')),sample_stdev_us=float(format(statistics.stdev(samples),'.12g'))))
        paired[variant]={(v['M'],v['K'],v['N'],v['policy']):v for v in rows}
        data[label]=rows
    for key,base in paired['control'].items():
        trial=paired['candidate'][key]
        assert base['operand_sha256']==trial['operand_sha256'] and base['output_sha256']==trial['output_sha256'],key
        table.append(dict(series=stem,M=key[0],K=key[1],N=key[2],policy=key[3],control_us=base['median_us'],candidate_us=trial['median_us'],
            control_p90_us=base['p90_us'],candidate_p90_us=trial['p90_us'],latency_change_pct=100*(trial['median_us']/base['median_us']-1),
            control_over_candidate=base['median_us']/trial['median_us']))
write_csv('comparisons.csv',table);write_csv('sample_statistics.csv',stats)
repeat=[]
for variant,stem in itertools.product(('control','candidate'),('series','fallback')):
    for x in data[variant+'_'+stem+'1']:
        y=next(v for v in data[variant+'_'+stem+'2'] if (v['M'],v['policy'])==(x['M'],x['policy']))
        assert x['operand_sha256']==y['operand_sha256'] and x['output_sha256']==y['output_sha256']
        repeat.append(dict(variant=variant,series=stem,M=x['M'],policy=x['policy'],first_us=x['median_us'],second_us=y['median_us'],change_pct=100*(y['median_us']/x['median_us']-1)))
write_csv('repeat_variability.csv',repeat)
primary=[v for v in table if v['series'].startswith('series')]
fallback=[v for v in table if v['series'].startswith('fallback')]
criteria=dict(no_flush_5pct_both_M_both_series=all(v['latency_change_pct']<=-5 for v in primary if v['policy']=='no_flush'),
    store_flush_regression_at_most_5pct=all(v['latency_change_pct']<=5 for v in primary if v['policy']=='store_flush'),
    fallback_no_reproducible_5pct_regression=not any(all(v['latency_change_pct']>5 for v in fallback if v['M']==m and v['policy']==p) for m,p in itertools.product((16,64),policies)))
profiles=[];units={'byte':1,'Kbyte':1000,'Mbyte':1e6,'Gbyte':1e9,'ns':.001,'us':1,'ms':1000}
for variant in ('control','candidate'):
    assert (r/(variant+'_profile.exit')).read_text().strip()=='0'
    assert hashes[variant] in (r/(variant+'_profile.log')).read_text()
    checked=[json.loads(x) for x in (r/(variant+'_profile.jsonl')).read_text().splitlines()]
    assert len(checked)==1 and checked[0]['num_bad']==0 and checked[0]['checked']==32768 and checked[0]['iters']==0 and checked[0]['warmup']==10
    raw=list(csv.reader((r/f'ncu/{variant}_M8_reused_kernel_replay_metrics.csv').open()))
    assert len(raw)==3 and len(raw[0])==len(raw[1])==len(raw[2])
    unit=dict(zip(raw[0],raw[1]));v=dict(zip(raw[0],raw[2]))
    assert v['Kernel Name']==build['changed_specializations'][0] and v['CC']=='12.0'
    out=dict(variant=variant,M=8,K=4096,N=4096,kernel=v['Kernel Name'])
    for metric in (r/'profile-metrics.txt').read_text().splitlines()+['launch__registers_per_thread_allocated','launch__shared_mem_per_block_allocated','profiler__replayer_passes']:
        value=float(v[metric].replace(',',''));assert math.isfinite(value)
        out[metric]=value*units.get(unit[metric].split('/')[0],1)
    assert out['launch__grid_size']==36 and out['launch__shared_mem_per_block_dynamic']==101376
    assert out['launch__registers_per_thread']==(126 if variant=='control' else 124)
    profiles.append(out)
assert checked[0]['output_sha256']==json.loads((r/'control_profile.jsonl').read_text())['output_sha256']
write_csv('profile_summary.csv',profiles)
for name in ('measurement-tooling.sha256','profile-tooling.sha256'):
    for line in (r/name).read_text().splitlines():
        sha,path=line.split(maxsplit=1)
        assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha,path
assert (r/'retained-project.log').read_text().count('ERROR SUMMARY: 0 errors')==6
assert (r/'dependency-diff.txt').read_text()==''
summary=dict(status='accepted' if all(criteria.values()) else 'rejected under predeclared performance criterion',criteria=criteria,
    baseline='fresh matched E006-corrected control, fixed host A',binary_sha256=hashes,
    timing_rows=sum(len(v) for v in data.values()),timed_samples=sum(len(v['samples_us']) for rows in data.values() for v in rows),
    smoke_rows=54,numerical_validation_rows=270,sanitizer_rows=162,profiles=2,full_output_bits_match=True,
    max_abs=max(v['max_abs'] for rows in data.values() for v in rows),max_rel_l2=max(v['rel_l2'] for rows in data.values() for v in rows),
    next_task='stop after the bounded E007 experiment and hand off the measured result')
(r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
