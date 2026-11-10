#!/usr/bin/env python3
"""E025 diagnostic: which operations carry the device time of an eager Marlin decode step?

profile.py OUT CONTEXT...  — per context: one warm request, then one profiled 8-token request (7 decode
steps). Device kernel time is attributed to the launching operator (self device time) for events that
start inside the decode span. Profiler overhead distorts host time; only device time is used.
"""
import collections
import json
import re
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402
from infer import Runner  # noqa: E402

I = D.I


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    work = I.Workload(D.E017 / 'results/e017_20261109', 'Q')
    summary = {}
    for context in [int(c) for c in sys.argv[2:]]:
        runner = Runner(work, 'Q', context)
        reference, _ = runner.request('primary', generated=8)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            row, _ = runner.request('primary', generated=8, markers=True)
        assert row['output_ids'] == reference['output_ids']
        events = prof.events()
        span = [e for e in events if e.name == 'E017/decode']
        assert len(span) == 1
        start, end = span[0].time_range.start, span[0].time_range.end
        device = lambda e: getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0)
        by_op, kernels = collections.Counter(), collections.Counter()
        for e in events:
            if not start <= e.time_range.start <= end:
                continue
            if str(e.device_type).endswith('CUDA'):
                kernels[e.name[:90]] += e.time_range.end - e.time_range.start
            elif device(e):
                name = re.sub(r'^E017/linear/model\.layers\.\d+\.(\w+)\.(\w+)/M1$', r'marlin/\2', e.name)
                by_op[name] += device(e)
        steps = row['decode_forwards']
        summary[context] = dict(decode_steps=steps, device_us_per_step=sum(kernels.values()) / steps,
                                by_operator_us_per_step={k: v / steps for k, v in by_op.most_common(40)},
                                by_kernel_us_per_step={k: v / steps for k, v in kernels.most_common(40)},
                                attributed_us_per_step=sum(by_op.values()) / steps, workspace=work.operator.audit())
        print('PROFILED', context, flush=True)
    if 128 in summary and 2048 in summary:  # the context-proportional part
        a, b = summary[128]['by_operator_us_per_step'], summary[2048]['by_operator_us_per_step']
        summary['delta_2048_minus_128_us_per_step'] = dict(sorted(((k, b.get(k, 0) - a.get(k, 0)) for k in set(a) | set(b)), key=lambda x: -x[1]))
    I.save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
