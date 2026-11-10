#!/usr/bin/env python3
"""Small pure-Python checks for timeline unions, clipping and late launches."""
from timeline import merged, summarize
from analyze import gain, passes, traversal, checked_sample
import copy

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

assert traversal(1)[::-1] == traversal(2)
assert passes([5,6]) and not passes([5,4.99]) and not passes([6,float('nan')])
assert 9.99 < gain(100,90) < 10.01
ref = dict(input_ids=list(range(128)),output_ids=list(range(64)),input_sha256='fixture',eos_positions=[])
row = dict(context=128,generated=64,decode_forwards=63,cache_length=191,variant='A',
    **ref,eos_policy='fixed length, EOS recorded and not used to stop',
    preparation_s=1.,prefill_to_first_s=2.,ttft_s=3.,post_first_boundary_s=.1,
    decode_s=4.,output_s=.2,e2e_s=7.3,peak_allocated=100,peak_reserved=120,
    device_total=1024,cache_class='DynamicCache',static_capacity=None)
checked_sample(row,ref,128)
for field,value in [('generated',63),('decode_s',float('nan')),('cache_length',192),
                    ('output_ids',[0]*64),('e2e_s',7.),('peak_reserved',1024)]:
    bad = copy.deepcopy(row);bad[field] = value
    try:
        checked_sample(bad,ref,128)
    except AssertionError:
        pass
    else:
        raise AssertionError(field)
print('PASS reversed orders, fixed5% gate, timing decomposition and six invalid-sample checks')
