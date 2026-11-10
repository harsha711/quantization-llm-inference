#!/usr/bin/env python3
"""E024 analysis: analyze.py DIR CONTEXT... — per (context, passage, series) decode gains and the robustness criteria."""
import json
import sys
from pathlib import Path

root, contexts = Path(sys.argv[1]), [int(c) for c in sys.argv[2:]]
cells, ok_median, ok_tail = [], True, True
for context in contexts:
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('R', 'GR', 'Q', 'GQ')}
            assert all(len(x['rows']) == 23 for x in m.values())
            gain = lambda a, b: 100 * (1 - m[a]['decode_s'] / m[b]['decode_s'])
            row = dict(context=context, prompt=prompt, series=s, decode_ms={v: round(1e3 * m[v]['decode_s'], 3) for v in m},
                       decode_range_ms={v: [round(1e3 * m[v]['decode_min'], 3), round(1e3 * m[v]['decode_max'], 3)] for v in m},
                       e2e_ms={v: round(1e3 * m[v]['e2e_s'], 3) for v in m},
                       gain_pct={f'{a}_vs_{b}': round(gain(a, b), 3) for a, b in (('GQ', 'R'), ('GQ', 'GR'), ('GQ', 'Q'), ('Q', 'R'), ('GR', 'R'))},
                       gq_slowest_below_fp16_fastest=m['GQ']['decode_max'] < min(m['R']['decode_min'], m['GR']['decode_min']))
            ok_median &= min(row['gain_pct']['GQ_vs_R'], row['gain_pct']['GQ_vs_GR']) >= 5
            ok_tail &= row['gq_slowest_below_fp16_fastest']
            cells.append(row)
worst = lambda k: min(c['gain_pct'][k] for c in cells)
best = lambda k: max(c['gain_pct'][k] for c in cells)
print(json.dumps(dict(cells=cells, criterion_a_median_5pct_everywhere=ok_median, criterion_b_no_overlap_everywhere=ok_tail,
                      gain_range_pct={k: [worst(k), best(k)] for k in cells[0]['gain_pct']}), indent=1))
