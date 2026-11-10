#!/usr/bin/env python3
"""E022 diagnostic: where does one M=1 decode step's projection time go? No optimization, no model run.

breakdown.py OUT REPS  — per rep, all 196 projections are called once back-to-back (as in a decode
step), then one synchronize. host = time until the last call returns, total = time after synchronize.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402

I = D.I


def main():
    out, reps = Path(sys.argv[1]), int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    torch.manual_seed(1)
    mods = []
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        q = I.Linear(name, r.bias, I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name]), 'Q', op).eval()
        x = torch.randn(1, 1, r.k, device='cuda', dtype=torch.float16)
        mods.append((r, q, x, x.reshape(1, r.k).contiguous(), torch.empty(1, r.n, device='cuda', dtype=torch.float16)))
    assert len(mods) == 196
    for r, q, x, flat, o in mods:  # the raw call must be the module's own computation
        q.operator.call(flat, q.qweight, q.scales, o)
        assert torch.equal(o if q.bias is None else o + q.bias, q(x).reshape(1, -1))
    paths = {
        'R_module': lambda r, q, x, flat, o: r(x),
        'Q_module': lambda r, q, x, flat, o: q(x),
        'Q_raw_call': lambda r, q, x, flat, o: q.operator.call(flat, q.qweight, q.scales, o),
        'Q_raw_call_sync_each': lambda r, q, x, flat, o: (q.operator.call(flat, q.qweight, q.scales, o), torch.cuda.synchronize()),
        'python_loop_only': lambda r, q, x, flat, o: None,
    }
    summary = dict(reps=reps, gpu=work.identity['gpu'], weight_bytes=dict(
        R=sum(r.weight.numel() * 2 for r, *_ in mods), Q=sum(q.qweight.numel() * q.qweight.element_size() + q.scales.numel() * 2 for _, q, *_ in mods)))
    raw = {}
    for order in (list(paths), list(paths)[::-1]):  # two reversed series
        for name in order:
            f = paths[name]
            rows = []
            for i in range(reps + 20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for m in mods:
                    f(*m)
                t1 = time.perf_counter()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                if i >= 20:
                    rows.append((t1 - t0, t2 - t0))
            raw.setdefault(name, []).append(rows)
    for name, series in raw.items():
        summary[name] = [dict(host_ms=1e3 * statistics.median(h for h, _ in rows), total_ms=1e3 * statistics.median(t for _, t in rows),
                              total_min_ms=1e3 * min(t for _, t in rows), total_max_ms=1e3 * max(t for _, t in rows)) for rows in series]
    summary['workspace'] = op.audit()
    I.save(out / 'raw.json', raw)
    I.save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
