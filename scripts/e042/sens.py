#!/usr/bin/env python3
"""E042: where does the plain-rounding block cost come from? sens.py OUT TEST VALID HEADHASH [WINDOWS]

Quality only. Every arm quantizes a SUBSET of the 196 block projections with the project quantizer (E039's rtn(), grid 15,
checked against E031's head hash), keeps the rest and the head FP16, and scores WikiText-2 valid and test with E032's
windows and metric (E039's score()). Arms:
  O, RTN                 controls (must reproduce E032 on test in a full run)
  T:<type>   x7          only that projection type, all 28 blocks
  B:<block>  x28         only that block, all 7 types
  KB3                    all quantized except the 3 blocks with the largest VALIDATION rise in the B pass (kept FP16)
  KT1                    all quantized except the type with the largest VALIDATION rise in the T pass (kept FP16)
Selection for KB3/KT1 reads the validation split only.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402  (sets up the E017 import path)
from quantize import tensor_sha  # noqa: E402

I, WINDOW, GROUP = D.I, 2048, 128
E032 = {'O': 8.605813234340317, 'RTN': 9.734986752455836}
TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
F = torch.nn.functional


def rtn(w):
    """E039 rtn() grid 15: w [N,K] FP32 -> FP16 reconstruction (project quantizer, scale max|w|/7, integers -7..7)."""
    g = w.view(w.shape[0], -1, GROUP)
    s = (g.abs().amax(2, keepdim=True) / 7).half().float()
    return ((torch.where(s > 0, g / s, 0).round().clamp(-7, 7) + 0.0) * s).half().view_as(w)


with torch.no_grad():
    out, (test, valid), head_hash = Path(sys.argv[1]), sys.argv[2:4], sys.argv[4]
    full = len(sys.argv) <= 5
    work = I.Workload(D.E017 / 'results/e017_20261109', 'O')
    work.model.requires_grad_(False)
    body = work.model.model
    tok = lambda text: work.tokenizer(text, add_special_tokens=False)['input_ids']
    raw = {name: Path(p).read_bytes() for name, p in (('test', test), ('valid', valid))}
    ids = {s: tok(raw[s].decode()) for s in ('valid', 'test')}
    nwin = {s: len(v) // WINDOW if full else int(sys.argv[5]) for s, v in ids.items()}

    fp16_head = work.model.lm_head.weight.detach().clone()
    assert tensor_sha(np.ascontiguousarray(rtn(fp16_head.float()).T.cpu().numpy())) == head_hash, 'rtn() is not the project quantizer'

    mods = {}
    for b, layer in enumerate(body.layers):
        for name, m in layer.named_modules():
            if isinstance(m, I.Linear):
                mods[(b, name.rsplit('.', 1)[-1])] = m
    nb = len(body.layers)
    assert len(mods) == 7 * nb == 196 and {t for _, t in mods} == set(TYPES)
    orig = {k: m.weight for k, m in mods.items()}
    quant = {k: torch.nn.Parameter(rtn(p.detach().float()), requires_grad=False) for k, p in orig.items()}
    total = sum(p.numel() for p in orig.values())

    arms = {}

    def score(tag, keys):
        keys = set(keys)
        for k, m in mods.items():
            m.weight = quant[k] if k in keys else orig[k]
        for split in ids:
            nll, correct = 0.0, 0
            for i in range(nwin[split]):
                x = torch.tensor(ids[split][i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
                logits = F.linear(body(input_ids=x, use_cache=False).last_hidden_state[0, :-1], fp16_head).float()
                assert torch.isfinite(logits).all()
                nll += F.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()
                correct += int((logits.argmax(-1) == x[0, 1:]).sum())
            n = nwin[split] * (WINDOW - 1)
            arms.setdefault(tag, {})[split] = dict(nll_sum=nll, perplexity=math.exp(nll / n), top1_accuracy_pct=100 * correct / n)
        arms[tag]['quantized_layers'] = len(keys)
        arms[tag]['quantized_param_fraction'] = sum(orig[k].numel() for k in keys) / total
        (out / 'arms.json').write_text(json.dumps(arms, indent=1))
        print('ARM', tag, {s: round(arms[tag][s]['perplexity'], 4) for s in ids}, flush=True)

    ppl = lambda a, s: arms[a][s]['perplexity']
    rise = lambda a, s: 100 * (ppl(a, s) / ppl('O', s) - 1)

    score('O', [])
    score('RTN', mods)
    if full:
        for a, p in E032.items():
            assert abs(ppl(a, 'test') / p - 1) < 1e-5, (a, ppl(a, 'test'), p)
    for t in TYPES:
        score(f'T:{t}', [k for k in mods if k[1] == t])
    for b in range(nb):
        score(f'B:{b}', [k for k in mods if k[0] == b])
    # confirmation arms: chosen from the validation split only
    worst_blocks = sorted(range(nb), key=lambda b: rise(f'B:{b}', 'valid'), reverse=True)[:3]
    worst_type = max(TYPES, key=lambda t: rise(f'T:{t}', 'valid'))
    score('KB3', [k for k in mods if k[0] not in worst_blocks])
    score('KT1', [k for k in mods if k[1] != worst_type])

    summary = {}
    for s in ids:
        r = rise('RTN', s)
        summary[s] = dict(
            rtn_rise_pct=r,
            sum_single_type_rise_pct=sum(rise(f'T:{t}', s) for t in TYPES),
            sum_single_block_rise_pct=sum(rise(f'B:{b}', s) for b in range(nb)),
            KB3_rise_pct=rise('KB3', s), KB3_recovered_fraction=1 - rise('KB3', s) / r,
            KT1_rise_pct=rise('KT1', s), KT1_recovered_fraction=1 - rise('KT1', s) / r)
    result = dict(sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}, windows=nwin,
                  scored_tokens={s: n * (WINDOW - 1) for s, n in nwin.items()}, gpu=work.identity['gpu'],
                  selection=dict(split='valid', worst_blocks=worst_blocks, worst_type=worst_type),
                  rise_pct={a: {s: rise(a, s) for s in ids} for a in arms}, summary=summary, arms=arms)
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(dict(selection=result['selection'], summary=summary)), flush=True)
