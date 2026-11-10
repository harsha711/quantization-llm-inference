#!/usr/bin/env python3
"""E023 analysis: analyze.py DIR — decode-median gains per context/series. Criterion: GQ >= 5% below Q in both series."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = {}
for context in (128, 2048):
    series = []
    for s in (1, 2):
        m = {v: json.loads((root / f'primary_{v}_{context}_s{s}.json').read_text()) for v in ('R', 'GR', 'Q', 'GQ')}
        assert all(len((root / f'primary_{v}_{context}_s{s}.jsonl').read_text().splitlines()) == 23 for v in m)
        gain = lambda a, b, k='decode_s': 100 * (1 - m[a][k] / m[b][k])
        series.append(dict(decode_ms={v: 1e3 * m[v]['decode_s'] for v in m}, e2e_ms={v: 1e3 * m[v]['e2e_s'] for v in m},
                           ttft_ms={v: 1e3 * m[v]['ttft_s'] for v in m}, peak_allocated={v: m[v]['peak_allocated'] for v in m},
                           decode_gain_pct=dict(GQ_vs_Q=gain('GQ', 'Q'), GQ_vs_R=gain('GQ', 'R'), GQ_vs_GR=gain('GQ', 'GR'),
                                                GR_vs_R=gain('GR', 'R'), Q_vs_R=gain('Q', 'R')),
                           e2e_gain_pct=dict(GQ_vs_R=gain('GQ', 'R', 'e2e_s'), GQ_vs_GR=gain('GQ', 'GR', 'e2e_s'))))
    out[context] = dict(series=series, passes=all(x['decode_gain_pct']['GQ_vs_Q'] >= 5 for x in series),
                        passes_vs_best_fp16=all(min(x['decode_gain_pct']['GQ_vs_R'], x['decode_gain_pct']['GQ_vs_GR']) >= 5 for x in series))
print(json.dumps(out, indent=1))
