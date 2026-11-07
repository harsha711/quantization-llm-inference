#!/usr/bin/env python3
"""Check the one conditional matched profile pair, keeping replay out of timing."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results',type=Path)
a=p.parse_args();r=a.results;repo=Path(__file__).resolve().parents[2]
summary=json.loads((r/'analysis.json').read_text());k,n=summary['profile_shape']
assert summary['timing_rows']==16 and summary['timed_samples']==1600
winner=next(v for v in summary['rectangles'] if (v['K'],v['N'])==(k,n))
assert winner['passes'] and all(x>=5 for x in winner['no_flush_gain_pct'])
assert (r/'profile-input-check.log').read_text().count(': OK')==5
assert (r/'post-profile-input-check.log').read_text().count(': OK')==5
assert (r/'e006-handoff-final-check.log').read_text().count(': OK')==113
assert (r/'metric-query.exit').read_text().strip()=='0'
query=gzip.decompress((r/'metric-query.log.gz').read_bytes())
assert hashlib.sha256(query).hexdigest()==(r/'metric-query-uncompressed.sha256').read_text().split()[0]
metrics=(r/'profile-metrics.txt').read_text().splitlines()
query_text=query.decode()
assert all(m+' ' in query_text for m in metrics if not m.startswith('launch__'))
for line in (r/'profile-tooling.sha256').read_text().splitlines():
    sha,path=line.split(maxsplit=1)
    assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha,path
units={'byte':1,'Kbyte':1000,'Mbyte':1000000,'Gbyte':1000000000,'ns':.001,'us':1,'ms':1000}
profiles=[];symbols=[]
for config in 'AC':
    assert (r/f'profile-{config}.exit').read_text().strip()=='0'
    assert summary['body_sha256'] in (r/f'profile-{config}.log').read_text()
    rows=[json.loads(v) for v in (r/f'profile-{config}.jsonl').read_text().splitlines()]
    assert len(rows)==1
    v=rows[0]
    assert (v['config'],v['M'],v['K'],v['N'],v['seed'],v['group'])==(config,4,k,n,2,128)
    assert v['num_bad']==0 and v['checked']==4*n and v['graph_check'] and v['repeated_launches']==3
    assert v['warmup']==10 and v['iters']==0 and v['policy']=='no_flush'
    assert all(v[x] for x in ('codes_match','scales_match','reconstruction_match','workspace_guards','lock_reset'))
    timed=[json.loads(x) for x in (r/'series1.jsonl').read_text().splitlines()]
    same=next(x for x in timed if (x['config'],x['K'],x['N'],x['policy'])==(config,k,n,'no_flush'))
    assert v['operand_sha256']==same['operand_sha256']
    tag=f'{config}_M4_K{k}_N{n}_reused_kernel_replay'
    raw=list(csv.reader((r/f'ncu/{tag}_metrics.csv').open()))
    assert len(raw)==3 and len(raw[0])==len(raw[1])==len(raw[2])
    unit=dict(zip(raw[0],raw[1]));values=dict(zip(raw[0],raw[2]))
    mangled=list(csv.DictReader((r/f'ncu/{tag}_mangled.csv').open()))
    assert len(mangled)==2 and mangled[1]['Kernel Name'].startswith('_ZN11e005_marlin6MarlinI')
    symbols.append(mangled[1]['Kernel Name'])
    assert 'Marlin<' in values['Kernel Name'] and ', 256, 1, 8, 8, 1, 4, 8, 0>' in values['Kernel Name']
    assert values['CC']=='12.0'
    out=dict(config=config,M=4,K=k,N=n,kernel=values['Kernel Name'])
    for m in metrics+['launch__registers_per_thread_allocated','launch__shared_mem_per_block_allocated','launch__occupancy_limit_shared_mem','launch__occupancy_limit_registers','profiler__replayer_passes']:
        value=float(values[m].replace(',',''));assert math.isfinite(value),m
        out[m]=value*units.get(unit[m].split('/')[0],1)
    assert out['launch__grid_size']==(36 if config=='A' else 72)
    assert out['launch__shared_mem_per_block_dynamic']==(101376 if config=='A' else 49664)
    assert out['launch__registers_per_thread']==126 and out['launch__block_size']==256
    profiles.append(out)
assert len(set(symbols))==1
assert profiles[0]['profiler__replayer_passes']==profiles[1]['profiler__replayer_passes']
assert profiles[0]['sm__inst_executed_pipe_tensor.sum']==profiles[1]['sm__inst_executed_pipe_tensor.sum']
with (r/'profile_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(profiles[0]),lineterminator='\n');w.writeheader();w.writerows(profiles)
report=dict(status='matched conditional profiles verified',profiles=2,shape=[4,k,n],
    replay_passes=profiles[0]['profiler__replayer_passes'],
    warp_instruction_change_pct=100*(profiles[1]['smsp__inst_executed.sum']/profiles[0]['smsp__inst_executed.sum']-1),
    L2_byte_change_pct=100*(profiles[1]['lts__t_bytes.sum']/profiles[0]['lts__t_bytes.sum']-1),
    automatic_dispatch_authorized=False)
(r/'profile-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
