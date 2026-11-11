#!/usr/bin/env python3
"""E033 analysis: analyze.py DIR — (1) vLLM+head versus ORT INT4 interleaved; (2) head gain in vLLM by batch size."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
load = lambda p, v, c, s: json.loads((root / f'{p}_{v}_{c}_s{s}.json').read_text())
tie, batches = {}, []
for c in (128, 2048):
    gains = []
    for p in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: load(p, v, c, s) for v in ('V', 'VH', 'OQ')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gains.append(dict(prompt=p, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                              VH_vs_OQ_pct=round(100 * (1 - m['VH']['decode_s'] / m['OQ']['decode_s']), 3),
                              VH_vs_V_pct=round(100 * (1 - m['VH']['decode_s'] / m['V']['decode_s']), 3)))
    g = [x['VH_vs_OQ_pct'] for x in gains]
    tie[c] = dict(cells=gains, outcome='faster' if min(g) >= 5 else 'slower' if max(g) <= -5 else 'tie' if max(map(abs, g)) < 5 else 'mixed')
    for b in (1, 2, 4, 8):
        for s in (1, 2):
            v, h = (load('primary', x if b == 1 else f'{x}b{b}', c, s) for x in ('V', 'VH'))
            batches.append(dict(context=c, batch=b, series=s, decode_ms=dict(V=round(1e3 * v['decode_s'], 3), VH=round(1e3 * h['decode_s'], 3)),
                                tokens_per_s=dict(V=round(63 * b / v['decode_s'], 1), VH=round(63 * b / h['decode_s'], 1)),
                                VH_vs_V_pct=round(100 * (1 - h['decode_s'] / v['decode_s']), 3),
                                tokens_agree_min=dict(V=v['tokens_agree_min'], VH=h['tokens_agree_min'])))
print(json.dumps(dict(vllm_head_versus_ort_int4=tie, batch_sweep=batches), indent=1))
