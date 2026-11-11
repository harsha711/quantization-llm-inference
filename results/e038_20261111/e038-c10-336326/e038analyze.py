#!/usr/bin/env python3
"""E038 analysis: analyze.py PLAINDIR CALDIR — decode medians of the calibrated checkpoint against the plain-rounded one, per cell."""
import json
import sys
from pathlib import Path

plain, cal = Path(sys.argv[1]), Path(sys.argv[2])
cells, worst, head_ok = [], {}, dict(ours=True, vllm=True)
for context in (128, 2048):
    for prompt in ('primary', 'heldout'):
        for s in (1, 2):
            m = {d: {v: json.loads((root / f'{prompt}_{v}_{context}_s{s}.json').read_text()) for v in ('GQA', 'GQAH', 'V', 'VH')} for d, root in (('plain', plain), ('cal', cal))}
            assert all(len(x['rows']) == 23 for d in m.values() for x in d.values())
            diff = {v: round(100 * (m['cal'][v]['decode_s'] / m['plain'][v]['decode_s'] - 1), 3) for v in m['cal']}  # + = calibrated slower
            gain = {d: dict(GQAH_vs_GQA=round(100 * (1 - m[d]['GQAH']['decode_s'] / m[d]['GQA']['decode_s']), 3),
                            VH_vs_V=round(100 * (1 - m[d]['VH']['decode_s'] / m[d]['V']['decode_s']), 3)) for d in m}
            cells.append(dict(context=context, prompt=prompt, series=s, decode_ms={d: {v: round(1e3 * x['decode_s'], 3) for v, x in m[d].items()} for d in m},
                              calibrated_vs_plain_pct=diff, head_gain_pct=gain,
                              vllm_tokens_agree_min={d: {v: m[d][v]['tokens_agree_min'] for v in ('V', 'VH')} for d in m}))
            for v, x in diff.items():
                worst[v] = max(worst.get(v, 0), abs(x))
            head_ok['ours'] &= gain['cal']['GQAH_vs_GQA'] >= 5
            head_ok['vllm'] &= gain['cal']['VH_vs_V'] >= 5
print(json.dumps(dict(cells=cells, max_abs_calibrated_vs_plain_pct=worst, equal_speed_within_5pct={v: x <= 5 for v, x in worst.items()},
                      calibrated_head_gain_at_least_5pct_all_cells=head_ok), indent=1))
