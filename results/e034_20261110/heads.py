#!/usr/bin/env python3
"""E034: perplexity of several output-head quantizers in ONE process. heads.py OUT TEXT HEADHASH [WINDOWS]

The 4-bit blocks (R) run once per window; every head variant is applied to the same final hidden states.
Variants (weight [K=1536, N=vocab], groups along K, symmetric, reconstruct FP16(q*s) with FP16 s):
  F      FP16 head (control: must reproduce E032's R)
  G128   E031's head: group 128, scale max/7 (control: must equal scripts/e017/quantize.py bit for bit, and E032's RH)
  G128M  group 128, scale = c*max/7 with c from SHRINK chosen per group to minimize squared weight error (same format as G128)
  G32    group 32, scale max/7        G32M  group 32 with the same search
  B8     group 128, 8-bit (scale max/127): upper bound, no runtime of ours executes it
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
from quantize import quantize  # noqa: E402

I, WINDOW = D.I, 2048
SHRINK = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
VARIANTS = dict(G128=(128, 7, [1.0]), G128M=(128, 7, SHRINK), G32=(32, 7, [1.0]), G32M=(32, 7, SHRINK), B8=(128, 127, [1.0]))


def requantize(w, group, qmax, shrink):
    """w [K,N] FP32 on the GPU -> reconstructed FP16 [K,N]; per (group, column) the candidate with least squared error."""
    k, n = w.shape
    g = w.view(k // group, group, n)
    amax = g.abs().amax(1, keepdim=True)
    best = err = None
    for c in shrink:
        s = (amax * c / qmax).half().float()
        rec = ((torch.where(s > 0, g / s, 0).round().clamp(-qmax, qmax) + 0.0) * s).half()  # + 0.0: a rounded -0 is code 0, i.e. +0, as in quantize.py
        e = (rec.float() - g).square().sum(1, keepdim=True)
        if best is None:
            best, err = rec, e
        else:
            better = e < err
            best, err = torch.where(better, rec, best), torch.where(better, e, err)
    return best.view(k, n)


with torch.inference_mode():
    out, text, head_hash = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    work = I.Workload(D.E017 / 'results/e017_20261109', 'R')
    original = work.model.lm_head.weight.detach()  # [N,K] FP16
    wt = original.float().T.contiguous()  # [K,N]
    heads, weight_error = dict(F=original), dict(F=0.0)
    for name, spec in VARIANTS.items():
        rec = requantize(wt, *spec)
        weight_error[name] = float((rec.float() - wt).norm() / wt.norm())
        heads[name] = rec.T.contiguous()
        del rec
    reference = quantize(np.ascontiguousarray(original.cpu().numpy().T))
    assert reference['hashes']['reconstructed'] == head_hash, 'not the head E031 timed'
    assert np.array_equal(heads['G128'].cpu().numpy().view(np.uint16), np.ascontiguousarray(reference['reconstructed'].T).view(np.uint16)), 'G128 differs from the project quantizer'
    del reference, wt
    raw = text.read_bytes()
    ids = work.tokenizer(raw.decode(), add_special_tokens=False)['input_ids']
    windows = int(sys.argv[4]) if len(sys.argv) > 4 else len(ids) // WINDOW
    nll, correct, top = {v: 0.0 for v in heads}, {v: 0 for v in heads}, {v: [] for v in heads}
    for i in range(windows):
        x = torch.tensor(ids[i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
        hidden = work.model.model(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
        for v, w in heads.items():
            logits = torch.nn.functional.linear(hidden, w).float()
            assert torch.isfinite(logits).all()
            nll[v] += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()
            top[v].append(logits.argmax(-1).cpu().numpy().astype(np.int32))
            correct[v] += int((logits.argmax(-1) == x[0, 1:]).sum())
    scored = windows * (WINDOW - 1)
    top = {v: np.concatenate(t) for v, t in top.items()}
    ppl = {v: math.exp(nll[v] / scored) for v in heads}
    label = lambda r: 'negligible' if r <= 1 else 'modest' if r <= 5 else 'substantial'
    result = dict(text_sha256=hashlib.sha256(raw).hexdigest(), windows=windows, scored_tokens=scored, gpu=work.identity['gpu'], shrink=SHRINK, variants={
        v: dict(weight_rel_l2=weight_error[v], nll_sum=nll[v], perplexity=ppl[v], top1_accuracy_pct=100 * correct[v] / scored,
                rise_vs_F_pct=100 * (ppl[v] / ppl['F'] - 1), label=label(100 * (ppl[v] / ppl['F'] - 1)),
                top1_agreement_with_F_pct=100 * float((top[v] == top['F']).mean())) for v in heads})
    assert all(1 < p < 1e4 for p in ppl.values())
    np.savez_compressed(out / 'top1.npz', **top)
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(result), flush=True)
