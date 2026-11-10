#!/usr/bin/env python3
"""Summarize observed timeline intervals without treating gaps as a speedup bound."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys


def merged(intervals):
    result = []
    for a,b in sorted(intervals):
        assert b >= a
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1],b)
        else:
            result.append([a,b])
    return result


def summarize(events, marker):
    spans = [e for e in events if e.get('name') == marker and e.get('ph') == 'X'
             and e.get('cat') == 'user_annotation']
    assert len(spans) == 1,(marker,len(spans))
    a,b = spans[0]['ts'],spans[0]['ts']+spans[0]['dur']
    assert b > a
    gpu = [e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')
           and e.get('ph') == 'X' and e['ts'] < b and e['ts']+e['dur'] > a]
    assert gpu
    busy = merged([(max(a,e['ts']),min(b,e['ts']+e['dur'])) for e in gpu])
    busy_us = sum(y-x for x,y in busy)
    gaps = [(busy[i][1],busy[i+1][0]) for i in range(len(busy)-1) if busy[i+1][0] > busy[i][1]]
    cpu = [e for e in events if a <= e.get('ts',-1) < b]
    runtime = [e for e in cpu if e.get('cat') in ('cuda_runtime','cuda_driver')]
    kernel_names = Counter(e['name'] for e in gpu if e.get('cat') == 'kernel')
    api_names = Counter(e['name'] for e in runtime)
    calls = {e.get('args',{}).get('correlation'):e for e in runtime if 'Launch' in e['name']}
    late = []
    ordered = sorted((e for e in gpu if e.get('cat') == 'kernel'),key=lambda e:e['ts'])
    for previous,current in zip(ordered,ordered[1:]):
        prev_end = previous['ts']+previous['dur']
        launch = calls.get(current.get('args',{}).get('correlation'))
        if launch and launch['ts'] > prev_end and current['ts'] > prev_end:
            late.append(current['ts']-prev_end)
    return dict(marker=marker,window_us=b-a,gpu_union_us=busy_us,
        gpu_uncovered_us=b-a-busy_us,gpu_busy_fraction=busy_us/(b-a),
        internal_gap_count=len(gaps),internal_gap_us=sum(y-x for x,y in gaps),
        max_internal_gap_us=max((y-x for x,y in gaps),default=0),
        late_cpu_launch_pairs=len(late),late_cpu_launch_gap_us=sum(late),
        kernels=sum(kernel_names.values()),top_kernels=kernel_names.most_common(15),
        runtime_calls=dict(api_names),
        cpu_alloc_events=sum('alloc' in e.get('name','').lower() or e.get('name') == '[memory]' for e in cpu),
        cpu_copy_events=sum('copy' in e.get('name','').lower() or 'memcpy' in e.get('name','').lower() for e in cpu),
        compiled_markers=sorted({e['name'] for e in cpu if any(s in e.get('name','') for s in
            ('Torch-Compiled','CompiledFunction','CUDA Graph','CUDAGraph'))}),
        graph_launch_calls=sum(v for name,v in api_names.items() if 'GraphLaunch' in name),
        gpu_streams=sorted({str(e.get('args',{}).get('stream')) for e in gpu}),
        note='Profiler-observed intervals. Gaps and API calls do not establish critical-path attribution or attainable speedup.')


if __name__ == '__main__':
    path = Path(sys.argv[1])
    if path.exists() and path.suffix != '.gz':
        events = json.loads(path.read_text())['traceEvents']
    else:
        with gzip.open(path if path.suffix == '.gz' else str(path)+'.gz','rt') as f:
            events = json.load(f)['traceEvents']
    result = [summarize(events,marker) for marker in ('E014/prefill','E014/decode')]
    print(json.dumps(result,indent=2))
