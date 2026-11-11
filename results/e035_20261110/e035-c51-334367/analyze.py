#!/usr/bin/env python3
"""E027 analysis: analyze.py DIR CONTEXT... — our paths versus vLLM per (context, passage, series)."""
import json
import sys
from pathlib import Path

root, contexts = Path(sys.argv[1]), [int(c) for c in sys.argv[2:]]
cells, verdict = [], {}
for context in contexts:
    gains = []
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('GR', 'GQ', 'GQA', 'V')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = lambda a, b, k='decode_s': round(100 * (1 - m[a][k] / m[b][k]), 3)
            cells.append(dict(context=context, prompt=prompt, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                              decode_range_ms={v: [round(1e3 * m[v]['decode_min'], 3), round(1e3 * m[v]['decode_max'], 3)] for v in m},
                              e2e_ms={v: round(1e3 * m[v]['e2e_s'], 3) for v in m}, vllm_first_ms=round(1e3 * m['V']['first_s'], 3),
                              vllm_tokens_agree_min=m['V']['tokens_agree_min'],
                              gain_pct=dict(GQA_vs_V=gain('GQA', 'V'), GQ_vs_V=gain('GQ', 'V'), V_vs_GR=gain('V', 'GR'), GQA_vs_GR=gain('GQA', 'GR'),
                                            e2e_GQA_vs_V=gain('GQA', 'V', 'e2e_s'))))
            gains.append(cells[-1]['gain_pct']['GQA_vs_V'])
    verdict[context] = 'faster' if min(gains) >= 5 else 'slower' if max(gains) <= -5 else 'comparable' if max(map(abs, gains)) < 5 else 'mixed'
print(json.dumps(dict(cells=cells, GQA_versus_vLLM_decode=verdict), indent=1))
