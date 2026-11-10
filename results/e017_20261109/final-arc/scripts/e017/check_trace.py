"""Exercise correlation, split launch and missing-coverage checks without CUDA."""
from trace_coverage import inspect
names={'layer':{'N':8960}}
events=[]
for phase,start,m in [('prefill',0,2048),('decode',100,1)]:
    events.append(dict(cat='user_annotation',name='E017/'+phase,ts=start,dur=90))
    for i in range(1 if phase=='prefill' else 7):
        ts=start+10*i+1
        events.append(dict(cat='user_annotation',name=f'E017/linear/layer/M{m}',ts=ts,dur=8))
        for j in range(2 if phase=='prefill' else 1):
            corr=len(events)
            events.append(dict(cat='cuda_runtime',name='cudaLaunchKernel',ts=ts+j+1,dur=.1,args={'correlation':corr}))
            events.append(dict(cat='kernel',name='void e005_marlin::Marlin<test>',ts=1000+corr,dur=.1,args={'correlation':corr}))
assert inspect(events,names,2048)['candidateprefillMarlinlaunches']==2
for remove in (0,1,2,3):
    changed=events.copy();changed.pop(remove)
    try:inspect(changed,names,2048)
    except AssertionError:continue
    raise AssertionError('accepted missing correlated coverage')
print('PASS runtime/kernel correlation, split-launch coverage and four missing-evidence cases')
