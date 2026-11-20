#!/usr/bin/env python3
"""E054 cross-node summary: summary.py RESULTS_DIR -> RESULTS_DIR/summary.json (per GPU type: R1/R2 per node and context,
decode/prefill ranges, H1, attribution, 5060 Ti node-to-node spread). Reads each full-*/analysis.json; no new statistics."""
import collections
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
types = collections.defaultdict(list)
for d in sorted(R.glob('full-*')):
    a = json.loads((d / 'analysis.json').read_text())
    gpu = (d / 'gpu.txt').read_text().splitlines()[0].split(',')[0].strip()
    types[gpu].append((d.name, a))
out = {}
for gpu, nodes in types.items():
    rows = []
    for name, a in nodes:
        for c, r in a['per_context'].items():
            g, p = r['GTA_vs_GQA_decode_ms'], r['GTA_vs_GQA_prefill_ms']
            x = a.get('attribution', {}).get(c, {})
            rows.append(dict(node=name, context=int(c), decode_pct_ours_below_marlin=[g['min'], g['max']], R1=g['outcome'],
                             R2=r['R2_GTA_vs_GQA']['holds'], stock_pct=[r['GT_vs_GQ_decode_ms']['min'], r['GT_vs_GQ_decode_ms']['max']],
                             H1=r['H1_pass'], ours_below_fp16_pct=[r['GTA_vs_GRA_decode_ms']['min'], r['GTA_vs_GRA_decode_ms']['max']],
                             prefill_pct_ours_below_marlin=[p['min'], p['max']], prefill=p['outcome'], median_ms=r['median_ms'],
                             gemm_diff_us=x.get('gemm_diff_us_per_step'), decode_diff_us=x.get('decode_diff_us_per_step'),
                             other_diff_us=x.get('other_diff_us_per_step'), attribution_within_25pct=x.get('within_25pct')))
    out[gpu] = dict(nodes=[n for n, _ in nodes], rows=rows,
                    R1_replicated=all(r['R1'] == 'comparable' for r in rows), R1_faster_anywhere=any(r['R1'] == 'faster' for r in rows),
                    R2_holds=f"{sum(r['R2'] for r in rows)}/{len(rows)}", H1_all=all(r['H1'] for r in rows),
                    decode_pct_range=[min(r['decode_pct_ours_below_marlin'][0] for r in rows), max(r['decode_pct_ours_below_marlin'][1] for r in rows)])
    if len(nodes) > 1:  # node-to-node spread of the per-node median ratio, per context
        out[gpu]['node_spread_pct_points'] = {c: round(max(m) - min(m), 3) for c in sorted({r['context'] for r in rows})
                                              for m in [[100 * (1 - r['median_ms']['GTA'] / r['median_ms']['GQA']) for r in rows if r['context'] == c]]}
(R / 'summary.json').write_text(json.dumps(out, indent=1))
for gpu, v in out.items():
    print(gpu, v['nodes'], 'R1 comparable everywhere:', v['R1_replicated'], 'faster anywhere:', v['R1_faster_anywhere'], 'R2:', v['R2_holds'],
          'H1:', v['H1_all'], 'decode % ours below Marlin:', v['decode_pct_range'], v.get('node_spread_pct_points', ''))
