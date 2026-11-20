#!/usr/bin/env python3
"""E056 analyzer: per-decode-step device time by kernel group for GTA (ours, tc_ladder), GQA (Marlin) and V (vLLM in-process).

analyze.py S  — reads S/prof-T/ours.json, S/prof-M/ours.json (E030 gprofile.py) and S/prof-V/vllm.json (E030 vprofile.py, whose
tokens were asserted equal to prof-V/ours.json, a copy of prof-M/ours.json). Writes JSON to stdout; raises on any completeness or
token failure. Grouping rules are the predeclared substring rules below (first match wins, in this order).
"""
import json
import sys
from pathlib import Path

GROUPS = [('gemm', ('Marlin', 'w4a16_tc')),
          ('head', ('gemvx',)),
          ('attention', ('gemmSN', 'softmax', 'SoftMax', 'flash_fwd', 'fmha', 'reshape_and_cache', 'index_copy'))]


def group(name):
    return next((g for g, keys in GROUPS if any(k in name for k in keys)), 'other')


def split(table):
    out = {g: dict(us=0.0, calls=0.0) for g in ('gemm', 'head', 'attention', 'other')}
    top = {g: [] for g in out}
    for name, (us, calls) in table.items():
        g = group(name)
        out[g]['us'] += us
        out[g]['calls'] += calls
        top[g].append((round(us, 1), round(calls, 2), name[:100]))
    for g in out:
        out[g] = {k: round(v, 2) for k, v in out[g].items()}
        out[g]['top'] = sorted(top[g], key=lambda r: -r[0])[:6]
    return out


S = Path(sys.argv[1])
T = json.loads((S / 'prof-T/ours.json').read_text())
M = json.loads((S / 'prof-M/ours.json').read_text())
V = json.loads((S / 'prof-V/vllm.json').read_text())
assert json.loads((S / 'prof-V/ours.json').read_text()) == M  # vprofile's token reference is the Marlin run
res = dict(contexts={})
for c in M:
    assert c in T and c in V, c
    # tokens: T equal to M (here); V equal to M (asserted inside vprofile.py before and during profiling)
    tokens_equal = T[c]['input_ids'] == M[c]['input_ids'] and T[c]['output_ids'] == M[c]['output_ids'] and len(M[c]['output_ids']) == 64
    assert tokens_equal, c
    arms = {}
    for arm, d in (('GTA', T[c]), ('GQA', M[c]), ('V', V[c])):
        t = d['kernel_us_and_calls_per_step']
        g = split(t)
        total = round(d['device_us_per_step'], 1)
        assert abs(sum(g[k]['us'] for k in g) - total) < 0.5
        arms[arm] = dict(total_us=total, launches=round(sum(v[1] for v in t.values()), 2), **g)
    gap = {}
    for a, b in (('GTA', 'V'), ('GQA', 'V'), ('GTA', 'GQA')):
        diff = {g: round(arms[a][g]['us'] - arms[b][g]['us'], 1) for g in ('gemm', 'head', 'attention', 'other')}
        tot = round(arms[a]['total_us'] - arms[b]['total_us'], 1)
        gap[f'{a}-{b}'] = dict(total_us=tot, by_group_us=diff,
                               share=({g: round(v / tot, 3) for g, v in diff.items()} if tot else None))
    res['contexts'][c] = dict(tokens_equal_T_M=tokens_equal, tokens_equal_V_M='asserted in vprofile.py', arms=arms, gap=gap)
print(json.dumps(res, indent=1))
for c, r in res['contexts'].items():
    for arm, a in r['arms'].items():
        print(f"{c:>5} {arm:4} total {a['total_us']:7.1f} launches {a['launches']:7.1f} | " +
              ' | '.join(f"{g} {a[g]['us']:7.1f} us {a[g]['calls']:6.1f}" for g in ('gemm', 'head', 'attention', 'other')), file=sys.stderr)
    for k, gp in r['gap'].items():
        print(f"{c:>5} gap {k}: {gp['total_us']} us, {gp['by_group_us']}", file=sys.stderr)
