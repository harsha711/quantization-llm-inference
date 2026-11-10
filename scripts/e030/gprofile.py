#!/usr/bin/env python3
"""E030 diagnostic: GPU kernel time per decode step of our graph-replayed Marlin path with direct attention (GQA).

gprofile.py OUT CONTEXT...  — per context: warm requests, then a profiled 2-token and a profiled 34-token request.
Per-kernel device time per decode step = (34-token total − 2-token total) / 32, which removes prefill.
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

spec = importlib.util.spec_from_file_location('att', Path(__file__).resolve().parent / 'att.py')
att = importlib.util.module_from_spec(spec)
spec.loader.exec_module(att)


def kernels(prof):
    total = collections.defaultdict(lambda: [0, 0])
    for e in prof.events():
        if str(e.device_type).endswith('CUDA') and not e.name.startswith('E017/'):  # annotations are mirrored on the device timeline
            total[e.name[:120]][0] += e.time_range.end - e.time_range.start
            total[e.name[:120]][1] += 1
    return total


def per_step(short, long, steps=32):
    rows = {k: [(long[k][0] - short[k][0]) / steps, (long[k][1] - short[k][1]) / steps] for k in set(short) | set(long)}
    return dict(sorted(rows.items(), key=lambda x: -x[1][0]))


if __name__ == '__main__':
    with torch.inference_mode():
        out = Path(sys.argv[1])
        out.mkdir(parents=True, exist_ok=False)
        att.ATT = True
        work = att.I.Workload(att.D.E017 / 'results/e017_20261109', 'Q')
        att.install(work.model)
        summary = {}
        for context in [int(c) for c in sys.argv[2:]]:
            runner = att.E.GraphRunner(work, 'Q', context)
            for _ in range(3):
                row, _ = runner.run('primary', True, generated=34)
            prof = {}
            for n in (2, 34):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                    r, _ = runner.run('primary', True, generated=n)
                assert r['output_ids'] == row['output_ids'][:n]
                prof[n] = kernels(p)
            table = per_step(prof[2], prof[34])
            summary[context] = dict(input_ids=row['input_ids'], output_ids=row['output_ids'], device_us_per_step=sum(v[0] for v in table.values()),
                                    kernel_us_and_calls_per_step=table)
            print('PROFILED', context, summary[context]['device_us_per_step'], flush=True)
        (out / 'ours.json').write_text(json.dumps(summary, indent=1))
