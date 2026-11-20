#!/usr/bin/env python3
"""E031: Marlin-quantized output head in our graph-replayed path.

head.py validate OUT CONTEXT...                       dual process; E021 gate on GQAH against an FP64 control that uses the same quantized head
head.py time OUT VARIANT CONTEXT PROMPT SERIES REFDIR one cell; VARIANT GQA (FP16 head) or GQAH (4-bit head through corrected Marlin)
The head weight is the tied FP16 embedding matrix quantized by the unchanged E017 quantizer; the embedding lookup stays FP16.
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('att', Path(__file__).resolve().parent / 'att.py')
att = importlib.util.module_from_spec(spec)
spec.loader.exec_module(att)
D, I, E, checked = att.D, att.I, att.E, att.checked
HEAD = 'O'  # O original FP16 head, Q Marlin, F64 reconstructed weights accumulated in FP64


def quantized_head(work):
    w = work.model.lm_head.weight.detach().cpu().numpy()  # [N,K] FP16
    data = I.repack(np.ascontiguousarray(w.T), *I.load_packer())
    return dict(data, K=w.shape[1], N=w.shape[0])


class Head(torch.nn.Module):
    def __init__(self, original, data, op, control):
        super().__init__()
        self.o, self.q = original, I.Linear('lm_head', None, data, 'Q', op).eval()
        if control:
            self.register_buffer('w64', torch.from_numpy(data['reconstructed']).cuda().double())  # [K,N]

    def forward(self, x):
        if HEAD == 'O':
            return self.o(x)
        if HEAD == 'Q':
            return self.q(x)
        assert HEAD == 'F64'
        return (x.reshape(-1, x.shape[-1]).double() @ self.w64).to(x.dtype).reshape(*x.shape[:-1], -1)


RUNS = (('F64', 'F64', False, False, 'O'), ('F64H', 'F64', False, False, 'F64'), ('GQA', 'Q', True, True, 'O'), ('GQAH', 'Q', True, True, 'Q'),
        ('AQH', 'Q', False, True, 'Q'), ('GQAH2', 'Q', True, True, 'Q'))
SAVED = ('GQA', 'GQAH')


def validate(out, contexts):
    global HEAD
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    att.install(work.model)
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        q = I.Linear(name, r.bias, I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name]), 'Q', op).eval()
        parent, leaf = name.rsplit('.', 1)
        setattr(work.model.get_submodule(parent), leaf, D.Dual(r, q))
    data = quantized_head(work)
    work.model.lm_head = Head(work.model.lm_head, data, op, True)
    verdict = dict(gpu=work.identity['gpu'], head_hashes=data['hashes'], cases={})
    for context in contexts:
        runner = E.GraphRunner(work, 'R', context)
        for prompt in ('primary', 'heldout'):
            L, rows = {}, {}
            for label, mode, graph, a, head in RUNS:
                D.MODE, att.ATT, HEAD = mode, a, head
                rows[label], logits = runner.run(prompt, graph, key=f'{mode}{a}{head}', generated=64, capture_logits=True)
                L[label] = logits.numpy().copy()
            n, tokens_ok, flip = E.prefix(L['GQAH'], L['F64H'])
            try:
                checked(rows['GQAH'], rows['GQAH'], context)
                state = True
            except AssertionError as error:
                state = repr(error)
            agree = lambda a, b: next((i for i, (x, y) in enumerate(zip(rows[a]['output_ids'], rows[b]['output_ids'])) if x != y), 64)
            m = agree('F64H', 'F64')  # steps before the head change alters the sequence; logits are comparable only there
            s = dict(compared_steps=n, flip=flip, gqah_vs_f64h_rel_l2=E.rel(L['GQAH'], L['F64H'], n), workspace=op.audit(),
                     head_effect=dict(leading_tokens_equal_control=m, leading_tokens_equal_marlin=agree('GQAH', 'GQA'),
                                      rel_l2_over_equal_steps=E.rel(L['F64H'], L['F64'], m) if m else None,
                                      rel_l2_first_step=E.rel(L['F64H'], L['F64'], 1)))
            s['criteria'] = dict(finite=bool(np.isfinite(L['GQAH']).all()), tokens=tokens_ok, rel_l2_vs_F64H=s['gqah_vs_f64h_rel_l2'] <= .003,
                                 graph_bitwise_eager=bool(np.array_equal(L['GQAH'], L['AQH'])), repeat_bitwise=bool(np.array_equal(L['GQAH'], L['GQAH2'])),
                                 workspace=all(s['workspace'].values()), state=state is True)
            s['accepted'] = all(s['criteria'].values())
            verdict['cases'][f'{context}-{prompt}'] = s
            np.savez_compressed(out / f'anchor-{context}-{prompt}.npz', **{v: L[v] for v in SAVED})
            I.save(out / f'reference-{context}-{prompt}.json', {v: dict(input_ids=rows[v]['input_ids'], output_ids=rows[v]['output_ids']) for v in SAVED})
            print(context, prompt, 'ACCEPTED' if s['accepted'] else 'REJECTED', json.dumps(s), flush=True)
    verdict['accepted'] = all(c['accepted'] for c in verdict['cases'].values())
    I.save(out / 'verdict.json', verdict)
    print('ALL ACCEPTED' if verdict['accepted'] else 'NOT ALL ACCEPTED', flush=True)
    assert verdict['accepted']


def timed(out, variant, context, prompt, series, refdir):
    global HEAD
    att.ATT = True
    work = I.Workload(D.E017 / 'results/e017_20261109', 'Q')
    att.install(work.model)
    if variant == 'GQAH':
        work.model.lm_head = Head(work.model.lm_head, quantized_head(work), work.operator, False)
        HEAD = 'Q'
    runner = E.GraphRunner(work, 'Q', context)
    reference = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[variant]
    row, logits = runner.run(prompt, True, capture_logits=True)
    checked(row, reference, context)
    assert np.array_equal(logits.numpy(), np.load(refdir / f'anchor-{context}-{prompt}.npz')[variant])
    rows = []
    for index in range(23):  # 3 warmups + 20 samples
        row, _ = runner.run(prompt, True)
        checked(row, reference, context)
        rows.append({k: row[k] for k in ('decode_s', 'ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved')} | dict(sample=index - 3))
    kept = rows[3:]
    summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'ttft_s', 'e2e_s')}
    summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
                   peak_allocated=max(r['peak_allocated'] for r in kept), gpu=work.identity['gpu'], workspace=work.operator.audit())
    assert all(summary['workspace'].values())
    I.save(out / f'{prompt}_{variant}_{context}_s{series}.json', summary)
    print('DONE', prompt, variant, context, series, summary['decode_s'], flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        target = Path(sys.argv[2])
        if sys.argv[1] == 'validate':
            target.mkdir(parents=True, exist_ok=False)
            validate(target, [int(c) for c in sys.argv[3:]])
        else:
            assert sys.argv[1] == 'time' and sys.argv[3] in SAVED
            timed(target, sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), Path(sys.argv[7]))
