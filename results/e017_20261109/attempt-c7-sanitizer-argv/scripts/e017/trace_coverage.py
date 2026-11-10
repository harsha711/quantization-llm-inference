"""Join actual corrected-Marlin kernels to recorded selected-layer CUDA launches."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def inspect(events,names,context):
    spans={}
    for phase in ('prefill','decode'):
        matches=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='E017/'+phase]
        assert len(matches)==1,(phase,len(matches));spans[phase]=matches[0]
    runtime={e.get('args',{}).get('correlation'):e for e in events if e.get('cat')=='cuda_runtime' and e.get('name','').startswith('cudaLaunch')}
    marlin=[e for e in events if e.get('cat')=='kernel' and 'Marlin<' in e.get('name','')]
    assert marlin and all('e005_marlin::Marlin<' in e['name'] for e in marlin)
    assert not any('GraphLaunch' in e.get('name','') for e in events if e.get('cat')=='cuda_runtime')
    counts={};descriptors={};assigned=[]
    for phase,span in spans.items():
        inside=lambda e:span['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=span['ts']+span['dur']+.001
        markers=[e for e in events if e.get('cat')=='user_annotation' and e.get('name','').startswith('E017/linear/') and inside(e)]
        parsed=Counter()
        kernels=[]
        for marker in markers:
            name,m=marker['name'].removeprefix('E017/linear/').rsplit('/M',1);m=int(m)
            assert name in names and m==(context if phase=='prefill' else 1)
            parsed[name]+=1
            calls=[e for e in marlin if (launch:=runtime.get(e.get('args',{}).get('correlation'))) is not None and marker['ts']<=launch['ts'] and launch['ts']+launch.get('dur',0)<=marker['ts']+marker['dur']+.001]
            expected=2 if m==2048 and names[name]['N']>4096 else 1
            assert len(calls)==expected,(phase,name,m,len(calls),expected)
            kernels.extend(calls)
            descriptors.setdefault(name,[]).extend(dict(phase=phase,M=m,name=e['name'],grid=e['args'].get('grid'),block=e['args'].get('block')) for e in calls)
        assert parsed==Counter({name:1 if phase=='prefill' else 7 for name in names}),parsed
        counts[phase]=len(kernels);assigned.extend(e['args']['correlation'] for e in kernels)
    assert len(assigned)==len(marlin) and len(set(assigned))==len(marlin),'missing or aliased launch correlation'
    return dict(passed=True,modulecoverage=len(names),candidateprefillMarlinlaunches=counts['prefill'],decode=counts['decode'],
        attention_kernels=sum(e.get('cat')=='kernel' and 'fmha_cutlassF_f16' in e.get('name','') for e in events),
        descriptors=descriptors,interpretation='actual runtime-to-kernel correlations, not complete arithmetic-equivalence proof')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--context',type=int,choices=(128,2048),required=True);a=p.parse_args()
    names={r['name']:r for r in json.loads((a.root/'artifacts/manifest.json').read_text())['coverage']}
    events=json.load(gzip.open(a.root/f'trace_Q_{a.context}-trace.json.gz'))['traceEvents']
    result=inspect(events,names,a.context)
    reference=json.load(gzip.open(a.root/f'trace_R_{a.context}-trace.json.gz'))['traceEvents']
    assert not any(e.get('cat')=='kernel' and 'Marlin<' in e.get('name','') for e in reference)
    assert result['attention_kernels']==224
    assert sum(e.get('cat')=='kernel' and 'fmha_cutlassF_f16' in e.get('name','') for e in reference)==224
    path=a.root/f'trace-coverage-{a.context}.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',a.context,result['modulecoverage'],result['candidateprefillMarlinlaunches'],result['decode'])
