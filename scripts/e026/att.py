#!/usr/bin/env python3
"""E026: single-query grouped decode attention (no key/value head expansion, FP32 scores) under graph replay.

att.py validate OUT CONTEXT...                             dual process; E021 gate on graph-Marlin with the new attention
att.py time OUT VARIANT CONTEXT PROMPT SERIES REFDIR       one cell; VARIANT in GR, GQ (stock attention), GRA, GQA (new)
Prefill and every multi-query call keep the stock SDPA path.
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
ATT = False  # read when a step is executed or captured; a captured graph keeps the choice it was captured with


def grouped(query, key, value, mask, scaling):
    """softmax(scaling * q k^T + mask) v for one query token, key/value heads shared by groups of query heads."""
    b, h, _, d = query.shape
    kvh, length = key.shape[1], key.shape[2]
    assert b == 1 and h % kvh == 0
    q = query.float().reshape(kvh, h // kvh, d)
    bias = mask[0, :, :, :length].float().expand(kvh, h // kvh, length)
    weights = torch.softmax(torch.baddbmm(bias, q, key[0].float().transpose(1, 2), alpha=scaling), -1)
    return torch.bmm(weights, value[0].float()).to(query.dtype).reshape(1, 1, h, d)


def install(model):
    import transformers.models.qwen2.modeling_qwen2 as M
    assert model.config._attn_implementation == 'sdpa'
    stock = M.ALL_ATTENTION_FUNCTIONS['sdpa']

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
        if not ATT or query.shape[2] != 1:
            return stock(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
        assert dropout == 0.0 and attention_mask is not None
        return grouped(query, key, value, attention_mask, scaling), None

    M.ALL_ATTENTION_FUNCTIONS = {'sdpa': attention}
    # self-check against the stock definition with expanded heads, FP32
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(1, 12, 1, 16, generator=g), torch.randn(1, 2, 40, 16, generator=g), torch.randn(1, 2, 40, 16, generator=g)
    mask = torch.zeros(1, 1, 1, 40)
    mask[..., 30:] = torch.finfo(torch.float16).min
    want = torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(6, 1), v.repeat_interleave(6, 1), attn_mask=mask, scale=.25)
    assert torch.allclose(grouped(q, k, v, mask, .25), want.transpose(1, 2), atol=1e-5, rtol=1e-5)


RUNS = (('F64', 'F64', False, False), ('GR', 'R', True, False), ('GQ', 'Q', True, False), ('AQ', 'Q', False, True),
        ('GRA', 'R', True, True), ('GQA', 'Q', True, True), ('GQA2', 'Q', True, True))
SAVED = ('GR', 'GQ', 'GRA', 'GQA')


def validate(out, contexts):
    global ATT
    results = D.E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    install(work.model)
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
            for label, mode, graph, att in RUNS:
                D.MODE, ATT = mode, att
                rows[label], logits = runner.run(prompt, graph, key=f'{mode}{att}', generated=64, capture_logits=True)
                L[label] = logits.numpy().copy()
            n, tokens_ok, flip = E.prefix(L['GQA'], L['F64'])
            try:
                checked(rows['GQA'], rows['GQA'], context)
                state = True
            except AssertionError as error:
                state = repr(error)
            s = dict(compared_steps=n, flip=flip, gqa_vs_f64_rel_l2=E.rel(L['GQA'], L['F64'], n), workspace=op.audit(),
                     reported=dict(gq_vs_f64_rel_l2=E.rel(L['GQ'], L['F64'], n), gra_vs_f64_rel_l2=E.rel(L['GRA'], L['F64'], n),
                                   gqa_vs_gq_rel_l2=E.rel(L['GQA'], L['GQ'], n), gra_vs_gr_rel_l2=E.rel(L['GRA'], L['GR'], n),
                                   tokens_equal_f64={k: rows[k]['output_ids'] == rows['F64']['output_ids'] for k in rows}))
            s['criteria'] = dict(finite=bool(np.isfinite(L['GQA']).all()), tokens=tokens_ok, rel_l2_vs_F64=s['gqa_vs_f64_rel_l2'] <= .003,
                                 graph_bitwise_eager=bool(np.array_equal(L['GQA'], L['AQ'])), repeat_bitwise=bool(np.array_equal(L['GQA'], L['GQA2'])),
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
    global ATT
    base, ATT = variant[1], variant.endswith('A')
    work = I.Workload(D.E017 / 'results/e017_20261109', base)
    install(work.model)
    runner = E.GraphRunner(work, base, context)
    reference = json.loads((refdir / f'reference-{context}-{prompt}.json').read_text())[variant]
    row, logits = runner.run(prompt, True, capture_logits=True)
    checked(row, reference, context)
    assert np.array_equal(logits.numpy(), np.load(refdir / f'anchor-{context}-{prompt}.npz')[variant])
    rows = []
    for index in range(23):  # 3 warmups + 20 samples
        row, _ = runner.run(prompt, True)
        checked(row, reference, context)
        rows.append({k: row[k] for k in ('decode_s', 'ttft_s', 'e2e_s', 'peak_allocated', 'peak_reserved')} | dict(sample=index - 3))
    work.weights_unchanged()
    kept = rows[3:]
    summary = {k: statistics.median(r[k] for r in kept) for k in ('decode_s', 'ttft_s', 'e2e_s')}
    summary.update(decode_min=min(r['decode_s'] for r in kept), decode_max=max(r['decode_s'] for r in kept), rows=rows,
                   peak_allocated=max(r['peak_allocated'] for r in kept), gpu=work.identity['gpu'],
                   workspace=work.operator.audit() if work.operator else None)
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
