#!/usr/bin/env python3
"""Small pure-Python checks for timeline unions, clipping and late launches."""
from timeline import merged, summarize

assert merged([(4,8),(1,3),(2,5),(9,10)]) == [[1,8],[9,10]]
events = [dict(name='E014/decode',cat='user_annotation',ph='X',ts=0,dur=20),
          dict(name='a',cat='kernel',ph='X',ts=-1,dur=5,args={'correlation':1,'stream':7}),
          dict(name='b',cat='kernel',ph='X',ts=10,dur=12,args={'correlation':2,'stream':7}),
          dict(name='cudaLaunchKernel',cat='cuda_runtime',ph='X',ts=8,dur=1,args={'correlation':2}),
          dict(name='cudaDeviceSynchronize',cat='cuda_runtime',ph='X',ts=19,dur=1)]
s = summarize(events,'E014/decode')
assert summarize(events+[dict(name='E014/decode',cat='gpu_user_annotation',ph='X',ts=0,dur=20)],'E014/decode') == s
assert s['gpu_union_us'] == 14 and s['gpu_uncovered_us'] == 6
assert s['internal_gap_count'] == 1 and s['internal_gap_us'] == 6
assert s['late_cpu_launch_pairs'] == 1 and s['late_cpu_launch_gap_us'] == 6
assert s['graph_launch_calls'] == 0 and s['runtime_calls']['cudaDeviceSynchronize'] == 1
try:
    summarize(events+events[:1],'E014/decode')
except AssertionError:
    pass
else:
    raise AssertionError('duplicate marker accepted')
print('PASS interval union/clipping, launch-gap accounting and duplicate-marker rejection')
