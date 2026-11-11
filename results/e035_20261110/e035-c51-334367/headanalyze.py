#!/usr/bin/env python3
"""E031 analysis: analyze.py DIR CONTEXT... — effect of the 4-bit output head in our path and in vLLM."""
import json
import sys
from pathlib import Path

root, contexts = Path(sys.argv[1]), [int(c) for c in sys.argv[2:]]
cells, passes = [], {}
for context in contexts:
    ok = dict(ours=True, vllm=True)
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('GQA', 'GQAH', 'V', 'VH')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = lambda a, b: round(100 * (1 - m[a]['decode_s'] / m[b]['decode_s']), 3)
            cells.append(dict(context=context, prompt=prompt, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                              decode_range_ms={v: [round(1e3 * m[v]['decode_min'], 3), round(1e3 * m[v]['decode_max'], 3)] for v in m},
                              e2e_ms={v: round(1e3 * m[v]['e2e_s'], 3) for v in m},
                              vllm_tokens_agree_min=dict(V=m['V']['tokens_agree_min'], VH=m['VH']['tokens_agree_min']),
                              gain_pct=dict(GQAH_vs_GQA=gain('GQAH', 'GQA'), VH_vs_V=gain('VH', 'V'), GQAH_vs_V=gain('GQAH', 'V'), GQAH_vs_VH=gain('GQAH', 'VH'))))
            ok['ours'] &= cells[-1]['gain_pct']['GQAH_vs_GQA'] >= 5
            ok['vllm'] &= cells[-1]['gain_pct']['VH_vs_V'] >= 5
    passes[context] = ok
print(json.dumps(dict(cells=cells, passes_5pct_all_cells=passes), indent=1))
