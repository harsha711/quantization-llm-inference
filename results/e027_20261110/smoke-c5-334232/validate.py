#!/usr/bin/env python3
"""E021 acceptance: corrected-Marlin Q judged against the FP64-projection control, not R's greedy tokens.

validate.py OUT GEN CONTEXT...   (reuses the E018 dual modules and modes)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402
from infer import Runner, checked  # noqa: E402

I = D.I


def prefix(q, f):
    """Steps to compare and whether token agreement holds: identical tokens, or a first difference
    at a step where the control's own logits for the two tokens are within E017's elementwise bound."""
    q, f = q.astype(np.float64).reshape(len(q), -1), f.astype(np.float64).reshape(len(f), -1)
    tq, tf = q.argmax(1), f.argmax(1)
    diff = np.flatnonzero(tq != tf)
    if not len(diff):
        return len(q), True, None
    i = int(diff[0])
    a, b = f[i, tf[i]], f[i, tq[i]]
    return i + 1, bool(abs(a - b) <= .03 + .01 * abs(a)), dict(step=i, control_token=int(tf[i]), q_token=int(tq[i]), control_margin=float(a - b))


def rel(c, r, n):
    c, r = c[:n].astype(np.float64).ravel(), r[:n].astype(np.float64).ravel()
    return float(np.linalg.norm(c - r) / np.linalg.norm(r))


def self_test():
    f = np.zeros((3, 4)); f[:, 0] = 24; f[1, 1] = 23.9
    q = f.copy(); assert prefix(q, f) == (3, True, None)
    q[1, 1] = 24.1; assert prefix(q, f)[:2] == (2, True)        # near-tie flip: allowed, prefix ends there
    f[1, 1] = 20; q[1, 1] = 25; assert prefix(q, f)[:2] == (2, False)  # real disagreement: rejected


def main():
    self_test()
    out, gen, contexts = Path(sys.argv[1]), int(sys.argv[2]), [int(c) for c in sys.argv[3:]]
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
    verdict = dict(generated=gen, gpu=work.identity['gpu'], cases={})
    for context in contexts:
        runner = Runner(work, 'R', context)
        for prompt in ('primary', 'heldout'):
            runs = {}
            for label in ('R', 'Q', 'Q2', 'F32', 'F64', 'P'):
                D.MODE = 'Q' if label == 'Q2' else label
                D.PROBE.clear()
                for d in duals:
                    d.calls = 0
                row, logits = runner.request(prompt, generated=gen, capture_logits=True)
                assert all(d.calls == gen for d in duals)
                runs[label] = dict(row=row, logits=logits.numpy().copy())
            L = {k: v['logits'] for k, v in runs.items()}
            probe = list(D.PROBE)
            n, tokens_ok, flip = prefix(L['Q'], L['F64'])
            try:  # structural state invariants of the Q request (positions, cache, reset, storage)
                checked(runs['Q']['row'], runs['Q']['row'], context, gen)
                state = True
            except AssertionError as error:
                state = repr(error)
            s = dict(compared_steps=n, flip=flip, q_vs_f64_rel_l2=rel(L['Q'], L['F64'], n),
                     reported=dict(r_vs_f64_rel_l2_same_steps=rel(L['R'], L['F64'], n), f32_vs_f64_rel_l2_same_steps=rel(L['F32'], L['F64'], n),
                                   q_vs_r_rel_l2_all=rel(L['Q'], L['R'], gen), old_gate_q_vs_r_bad=D.gate(L['Q'], L['R'])['bad'],
                                   tokens_equal_F64={k: v['row']['output_ids'] == runs['F64']['row']['output_ids'] for k, v in runs.items()},
                                   R_prefix=prefix(L['R'], L['F64']), F32_prefix=prefix(L['F32'], L['F64']),
                                   R_bad_calls=sum(bool(p['R']['bad']) for p in probe),
                                   Q_max_rel_l2=max(p['Q']['rel_l2'] for p in probe), checked=sum(p['checked'] for p in probe),
                                   max_gpu_vs_cpu_fp64_rel=max(p.get('gpu_vs_cpu_fp64_rel', 0) for p in probe)),
                     failing=[p for p in probe if p['Q']['bad']][:50], workspace=op.audit())
            s['criteria'] = dict(
                finite=bool(np.isfinite(L['Q']).all()), tokens=tokens_ok,
                deterministic=bool(np.array_equal(L['Q'], L['Q2']) and np.array_equal(L['P'], L['R'])),
                operators=len(probe) == 196 * gen and not s['failing'], workspace=all(s['workspace'].values()),
                rel_l2_vs_F64=s['q_vs_f64_rel_l2'] <= .003, state=state is True)
            s['accepted'] = all(s['criteria'].values())
            verdict['cases'][f'{context}-{prompt}'] = s
            I.save(out / f'probe-{context}-{prompt}.json', probe)
            np.savez_compressed(out / f'logits-{context}-{prompt}.npz', **{k: L[k] for k in ('R', 'Q', 'F32', 'F64')})
            print(context, prompt, 'ACCEPTED' if s['accepted'] else 'REJECTED', json.dumps({k: v for k, v in s.items() if k != 'failing'}), flush=True)
    verdict['accepted'] = all(c['accepted'] for c in verdict['cases'].values())
    I.save(out / 'verdict.json', verdict)
    print('ALL ACCEPTED' if verdict['accepted'] else 'NOT ALL ACCEPTED', flush=True)
    assert verdict['accepted']


if __name__ == '__main__':
    with torch.inference_mode():
        main()
