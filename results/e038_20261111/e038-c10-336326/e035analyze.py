#!/usr/bin/env python3
"""E035 analysis: analyze.py DIR — vLLM decode with FP16 head (V), 4-bit head (VH) and 8-bit head (VH8), interleaved."""
import json
import sys
from pathlib import Path

root, cells = Path(sys.argv[1]), []
for c in (128, 2048):
    for p in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{p}_{v}_{c}_s{s}.json').read_text()) for v in ('V', 'VH', 'VH8')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = {v: 100 * (1 - m[v]['decode_s'] / m['V']['decode_s']) for v in ('VH', 'VH8')}
            cells.append(dict(context=c, prompt=p, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                              VH_vs_V_pct=round(gain['VH'], 3), VH8_vs_V_pct=round(gain['VH8'], 3), retained_pct=round(100 * gain['VH8'] / gain['VH'], 1),
                              tokens_agree_min={v: m[v]['tokens_agree_min'] for v in m}))
print(json.dumps(dict(outcome='pass' if min(x['VH8_vs_V_pct'] for x in cells) >= 5 else 'fail', cells=cells), indent=1))
