#!/usr/bin/env python3
"""E024 robustness of the E020/E023 gains: more contexts, both passages, optionally another node.

robust.py validate OUT CONTEXT...                              dual process, E021 gate on graph-Marlin, saves references
robust.py time OUT VARIANT CONTEXT PROMPT SERIES REFDIR        one cell; VARIANT in R, GR, Q, GQ
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location('e023_run', Path(__file__).resolve().parent / 'run.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)  # E023 tooling: GraphRunner, E021 gate helpers, E018 dual modules
D, I, checked = E.D, E.I, E.checked
VARIANTS = (('R', 'R', False), ('GR', 'R', True), ('Q', 'Q', False), ('GQ', 'Q', True))


def validate(out, contexts):
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        q = I.Linear(name, r.bias, I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name]), 'Q', op).eval()
        parent, leaf = name.rsplit('.', 1)
        setattr(work.model.get_submodule(parent), leaf, D.Dual(r, q))
    verdict = dict(gpu=work.identity['gpu'], cases={})
    for context in contexts:
        runner = E.GraphRunner(work, 'R', context)
        for prompt in ('primary', 'heldout'):
            L, rows = {}, {}
            for label, mode, graph in (('F64', 'F64', False),) + VARIANTS + (('GQ2', 'Q', True),):
                D.MODE = mode
                rows[label], logits = runner.run(prompt, graph, key=mode, generated=64, capture_logits=True)
                L[label] = logits.numpy().copy()
            n, tokens_ok, flip = E.prefix(L['GQ'], L['F64'])
            try:
                checked(rows['GQ'], rows['GQ'], context)
                state = True
            except AssertionError as error:
                state = repr(error)
            s = dict(compared_steps=n, flip=flip, gq_vs_f64_rel_l2=E.rel(L['GQ'], L['F64'], n), workspace=op.audit(),
                     reported=dict(gr_bitwise_r=bool(np.array_equal(L['GR'], L['R'])), r_tokens_equal_q=rows['R']['output_ids'] == rows['Q']['output_ids']))
            s['criteria'] = dict(finite=bool(np.isfinite(L['GQ']).all()), tokens=tokens_ok, rel_l2_vs_F64=s['gq_vs_f64_rel_l2'] <= .003,
                                 graph_bitwise_eager=bool(np.array_equal(L['GQ'], L['Q'])), repeat_bitwise=bool(np.array_equal(L['GQ'], L['GQ2'])),
                                 workspace=all(s['workspace'].values()), state=state is True)
            s['accepted'] = all(s['criteria'].values())
            verdict['cases'][f'{context}-{prompt}'] = s
            np.savez_compressed(out / f'anchor-{context}-{prompt}.npz', **{v: L[v] for v, _, _ in VARIANTS})
            I.save(out / f'reference-{context}-{prompt}.json', {v: dict(input_ids=rows[v]['input_ids'], output_ids=rows[v]['output_ids']) for v, _, _ in VARIANTS})
            print(context, prompt, 'ACCEPTED' if s['accepted'] else 'REJECTED', json.dumps(s), flush=True)
    verdict['accepted'] = all(c['accepted'] for c in verdict['cases'].values())
    I.save(out / 'verdict.json', verdict)
    print('ALL ACCEPTED' if verdict['accepted'] else 'NOT ALL ACCEPTED', flush=True)
    assert verdict['accepted']


def timed(out, variant, context, prompt, series, refdir):
    base, graph = variant[-1], variant.startswith('G')
    work = I.Workload(D.E017 / 'results/e017_20261109', base)
    runner = E.GraphRunner(work, base, context)
    reference = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[variant]  # this variant's own validated tokens
    row, logits = runner.run(prompt, graph, capture_logits=True)
    checked(row, reference, context)
    assert np.array_equal(logits.numpy(), np.load(refdir / f'anchor-{context}-{prompt}.npz')[variant])
    rows = []
    for index in range(23):  # 3 warmups + 20 samples
        row, _ = runner.run(prompt, graph)
        checked(row, reference, context)
        rows.append({k: row[k] for k in ('decode_s', 'ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved')} | dict(sample=index - 3))
    work.weights_unchanged()
    tag = f'{prompt}_{variant}_{context}_s{series}'
    kept = rows[3:]
    summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'ttft_s', 'e2e_s')}
    summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
                   peak_allocated=max(r['peak_allocated'] for r in kept), gpu=work.identity['gpu'],
                   workspace=work.operator.audit() if work.operator else None)
    I.save(out / (tag + '.json'), summary)
    print('DONE', tag, summary['decode_s'], flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        target = Path(sys.argv[2])
        if sys.argv[1] == 'validate':
            target.mkdir(parents=True, exist_ok=False)
            validate(target, [int(c) for c in sys.argv[3:]])
        else:
            assert sys.argv[1] == 'time' and sys.argv[3] in ('R', 'GR', 'Q', 'GQ')
            timed(target, sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), Path(sys.argv[7]))
