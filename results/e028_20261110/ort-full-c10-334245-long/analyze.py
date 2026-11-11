#!/usr/bin/env python3
"""E028 analysis: analyze.py DIR FP16VARIANT INT4VARIANT CONTEXT... — our GR/GQA versus one external stack."""
import json
import sys
from pathlib import Path

root, f, q, contexts = Path(sys.argv[1]), sys.argv[2], sys.argv[3], [int(c) for c in sys.argv[4:]]
cells, verdict = [], {}
for context in contexts:
    gains = []
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('GR', 'GQA', f, q)}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = lambda a, b, k='decode_s': round(100 * (1 - m[a][k] / m[b][k]), 3)
            cells.append(dict(context=context, prompt=prompt, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                              decode_range_ms={v: [round(1e3 * m[v]['decode_min'], 3), round(1e3 * m[v]['decode_max'], 3)] for v in m},
                              e2e_ms={v: round(1e3 * m[v]['e2e_s'], 3) for v in m}, tokens_agree_min={v: m[v]['tokens_agree_min'] for v in (f, q)},
                              gain_pct={f'GQA_vs_{q}': gain('GQA', q), f'GR_vs_{f}': gain('GR', f), f'{q}_vs_{f}': gain(q, f),
                                        f'e2e_GQA_vs_{q}': gain('GQA', q, 'e2e_s')}))
            gains.append(cells[-1]['gain_pct'][f'GQA_vs_{q}'])
    verdict[context] = 'faster' if min(gains) >= 5 else 'slower' if max(gains) <= -5 else 'comparable' if max(map(abs, gains)) < 5 else 'mixed'
print(json.dumps(dict(cells=cells, GQA_versus_external_int4_decode=verdict), indent=1))
