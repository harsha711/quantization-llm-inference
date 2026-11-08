#!/usr/bin/env python3
"""Small dynamic-cache smoke followed by short full-model CPU/GPU timelines."""
import argparse
import json
from pathlib import Path

import torch

from model import Workload, Runner


def save(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    a = p.parse_args()
    work = Workload(a.results)
    save(a.results/'baseline-identity.json',work.identity)
    for length in (128,2048):
        runner = Runner(work,'A',length)
        first = runner.request('primary',generated=8,capture_logits=True)
        second = runner.request('primary',generated=8,capture_logits=True)
        assert first['output_ids'] == second['output_ids']
        assert torch.equal(first['logits'],second['logits'])
        assert torch.isfinite(first['logits']).all()
        assert second['peak_reserved'] < second['device_total']*0.75
        first.pop('logits');second.pop('logits')
        save(a.results/f'smoke_{length}.json',dict(first=first,second=second))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
            record = runner.request('primary',generated=8,markers=True)
        record.pop('logits')
        save(a.results/f'trace_request_A_{length}.json',record)
        prof.export_chrome_trace(str(a.results/f'timeline_A_{length}.json'))
        (a.results/f'operations_A_{length}.txt').write_text(prof.key_averages().table(
            sort_by='self_cuda_time_total',row_limit=30))
        print('PASS dynamic smoke/trace',length,record['output_ids'],flush=True)
        del runner
    work.weights_unchanged()
    print('PASS baseline repeated logits/tokens, memory margin, fixed weights',flush=True)
