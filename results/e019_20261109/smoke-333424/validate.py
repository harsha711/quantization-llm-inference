#!/usr/bin/env python3
"""E019 model gate: Q accepted only relative to Marlin-free controls. Reuses E018 modes."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402  (E018 diagnostic: Dual, gate, modes; adds E017 paths)
from infer import Runner, checked  # noqa: E402

I = D.I


def main():
    out, context, gen = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ref = Path(sys.argv[4]) if len(sys.argv) > 4 else None  # root with unchanged infer.py R validation
    out.mkdir(parents=True, exist_ok=False)
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    duals = []
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        q = I.Linear(name, r.bias, I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name]), 'Q', op).eval()
        parent, leaf = name.rsplit('.', 1)
        duals.append(D.Dual(r, q))
        setattr(work.model.get_submodule(parent), leaf, duals[-1])
    assert len(duals) == 196
    runner = Runner(work, 'R', context)
    verdict = dict(context=context, generated=gen, gpu=work.identity['gpu'], cases={})
    for index, prompt in enumerate(('primary', 'heldout')):
        runs = {}
        for label in ('R', 'Q', 'Q2', 'F32', 'F64', 'P'):
            D.MODE = 'Q' if label == 'Q2' else label
            D.PROBE.clear()
            for d in duals:
                d.calls = 0
            row, logits = runner.request(prompt, generated=gen, capture_logits=True)
            assert all(d.calls == gen for d in duals)
            runs[label] = dict(row=row, logits=logits.numpy().copy())
            print('RAN', context, prompt, label, flush=True)
        L = {k: v['logits'] for k, v in runs.items()}
        probe = list(D.PROBE)
        s = dict(tokens_equal_R={k: v['row']['output_ids'] == runs['R']['row']['output_ids'] for k, v in runs.items()},
                 q_finite=bool(np.isfinite(L['Q']).all()), q_repeat_bitwise=bool(np.array_equal(L['Q'], L['Q2'])),
                 p_bitwise_R=bool(np.array_equal(L['P'], L['R'])), workspace=op.audit(),
                 Q_vs_R=D.gate(L['Q'], L['R']), F32_vs_R=D.gate(L['F32'], L['R']), F64_vs_R=D.gate(L['F64'], L['R']),
                 R_vs_F64=D.gate(L['R'], L['F64']), Q_vs_F64=D.gate(L['Q'], L['F64']), F32_vs_F64=D.gate(L['F32'], L['F64']),
                 probe=dict(calls=len(probe), checked=sum(p['checked'] for p in probe),
                            Q_bad_calls=sum(bool(p['Q']['bad']) for p in probe), R_bad_calls=sum(bool(p['R']['bad']) for p in probe),
                            Q_median_rel_l2=float(np.median([p['Q']['rel_l2'] for p in probe])),
                            R_median_rel_l2=float(np.median([p['R']['rel_l2'] for p in probe])),
                            Q_max_rel_l2=max(p['Q']['rel_l2'] for p in probe), R_max_rel_l2=max(p['R']['rel_l2'] for p in probe),
                            max_gpu_vs_cpu_fp64_rel=max(p.get('gpu_vs_cpu_fp64_rel', 0) for p in probe),
                            failing=[p for p in probe if p['Q']['bad'] or p['R']['bad']][:50]))
        s['state_ok'] = s['fresh_R_bitwise_infer_R'] = None
        if ref is not None:  # tie this dual-mode process to the separate-process path used for timing
            tag = f'validation_R_{context}_{prompt}_{index}'
            s['fresh_R_bitwise_infer_R'] = bool(np.array_equal(L['R'], np.load(ref / (tag + '-output.npz'))['logits']))
            try:
                checked(runs['Q']['row'], json.loads((ref / (tag + '-output.json')).read_text()), context, gen)
                s['state_ok'] = True
            except AssertionError as error:
                s['state_ok'] = repr(error)
        floor = max(s['R_vs_F64']['rel_l2'], s['F32_vs_F64']['rel_l2'])
        s['criteria'] = dict(
            finite=s['q_finite'], tokens=all(s['tokens_equal_R'].values()), deterministic=s['q_repeat_bitwise'] and s['p_bitwise_R'],
            operators=s['probe']['Q_bad_calls'] == 0 and s['probe']['calls'] == 196 * gen,
            workspace=all(s['workspace'].values()), rel_l2_vs_R=s['Q_vs_R']['rel_l2'] <= .003,
            not_farther_from_F64_than_marlin_free=s['Q_vs_F64']['rel_l2'] <= floor,
            state=s['state_ok'] is True or ref is None, anchor=s['fresh_R_bitwise_infer_R'] is not False)
        s['accepted'] = all(s['criteria'].values())
        verdict['cases'][prompt] = s
        I.save(out / f'probe-{prompt}.json', probe)
        np.savez_compressed(out / f'logits-{prompt}.npz', **{k: L[k] for k in ('R', 'Q', 'F32', 'F64')})
    verdict['accepted'] = all(c['accepted'] for c in verdict['cases'].values())
    I.save(out / 'verdict.json', verdict)
    for prompt, c in verdict['cases'].items():
        print(prompt, json.dumps(dict(criteria=c['criteria'], **{k: {x: c[k][x] for x in ('bad', 'max_abs', 'rel_l2', 'centered_bad')}
              for k in ('Q_vs_R', 'F32_vs_R', 'F64_vs_R', 'R_vs_F64', 'Q_vs_F64', 'F32_vs_F64')},
              probe={k: v for k, v in c['probe'].items() if k != 'failing'})))
    print('ACCEPTED' if verdict['accepted'] else 'REJECTED', context, flush=True)
    assert verdict['accepted']


if __name__ == '__main__':
    with torch.inference_mode():
        main()
