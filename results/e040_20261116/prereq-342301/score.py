#!/usr/bin/env python3
"""E040: score AWQ arms with E032's windows and metric. score.py OUT TEST VALID ARM=NPZ... [WINDOWS]

Frozen E017 environment. O must reproduce E032 (full runs). Each NPZ (from awq.py) replaces the 196 block weights with
its reconstructed FP16 weights and copies its changed norm weights / biases; originals are restored between arms.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402
from quantize import tensor_sha  # noqa: E402

I, WINDOW, F = D.I, 2048, torch.nn.functional
E032_O = 8.605813234340317
E039 = {'L': 9.049102938610124, 'AT15': 9.089416695696391}  # E039 test perplexities

with torch.no_grad():
    out, test, valid = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    arms_in = [a.split('=', 1) for a in sys.argv[4:] if '=' in a]
    rest = [a for a in sys.argv[4:] if '=' not in a]
    full = not rest
    work = I.Workload(D.E017 / 'results/e017_20261109', 'O')
    model = work.model
    model.requires_grad_(False)
    raw = {'test': Path(test).read_bytes(), 'valid': Path(valid).read_bytes()}
    ids = {s: work.tokenizer(raw[s].decode(), add_special_tokens=False)['input_ids'] for s in ('valid', 'test')}
    nwin = {s: len(v) // WINDOW if full else int(rest[0]) for s, v in ids.items()}
    head = model.lm_head.weight.detach().clone()
    mods = dict(model.named_modules())
    arms = {}

    def score(tag):
        for split in ids:
            nll, correct = 0.0, 0
            for i in range(nwin[split]):
                x = torch.tensor(ids[split][i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
                logits = F.linear(model.model(input_ids=x, use_cache=False).last_hidden_state[0, :-1], head).float()
                assert torch.isfinite(logits).all()
                nll += F.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()
                correct += int((logits.argmax(-1) == x[0, 1:]).sum())
            n = nwin[split] * (WINDOW - 1)
            arms.setdefault(tag, {})[split] = dict(nll_sum=nll, perplexity=math.exp(nll / n), top1_accuracy_pct=100 * correct / n)
        (out / 'arms.json').write_text(json.dumps(arms, indent=1))
        print('ARM', tag, {s: round(arms[tag][s]['perplexity'], 4) for s in ids}, flush=True)

    score('O')
    if full:
        assert abs(arms['O']['test']['perplexity'] / E032_O - 1) < 1e-5
    meta = {}
    for tag, npz in arms_in:
        info = json.loads(Path(npz).with_suffix('.json').read_text())
        data, saved = np.load(npz), {}
        nlin = 0
        for key in data.files:
            prefix, attr = key.rsplit('.', 1)
            m = mods[prefix]
            cur = getattr(m, attr)
            val = torch.from_numpy(data[key]).cuda()
            assert cur is not None and cur.shape == val.shape and cur.dtype == torch.float16, key
            if isinstance(m, I.Linear) and attr == 'weight':
                assert info['original_sha256'][prefix] == tensor_sha(cur.detach().cpu().numpy()), ('different original weights', prefix)
                nlin += 1
            saved[key] = cur.detach().clone()
            cur.data.copy_(val)
        assert nlin == 196, nlin
        score(tag)
        for key, v in saved.items():
            prefix, attr = key.rsplit('.', 1)
            getattr(mods[prefix], attr).data.copy_(v)
        meta[tag] = {k: v for k, v in info.items() if k not in ('original_sha256', 'linears')}
        del data, saved
    score('O_restored')  # originals restored after every arm
    assert all(arms['O_restored'][s]['nll_sum'] == arms['O'][s]['nll_sum'] for s in ids), 'restore failed'

    ppl = lambda a: arms[a]['test']['perplexity']
    label = lambda r: 'negligible' if r <= 1 else 'modest' if r <= 5 else 'substantial'
    compare = {}
    for tag, _ in arms_in:
        r = {s: 100 * (arms[tag][s]['perplexity'] / arms['O'][s]['perplexity'] - 1) for s in ids}
        compare[tag] = dict(rise_vs_O=r, label=label(r['test']), vs_E039_L_pct=100 * (ppl(tag) / E039['L'] - 1))
    result = dict(sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}, windows=nwin, gpu=work.identity['gpu'],
                  e039=E039, arms=arms, compare=compare, awq=meta)
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(compare), flush=True)
