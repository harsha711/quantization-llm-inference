#!/usr/bin/env python3
"""E039: is the remaining block cost our quantizer or the model? grid.py OUT TEST VALID C4 HEADHASH NCAL [WINDOWS]

E037's code (scripts/e037/calib.py) with the grid as a parameter; S, selection and the head removed. Grids:
  15  the project contract: scale max|w|/7, integers -7..7 (E017-E038)
  16  reference GPTQ / llm-compressor W4A16: scale 2*max|w|/15, integers -8..7 (not the runtime contract; quality only)
Arms (test and validation, FP16 head, E032's windows and metric): O, RTN (controls, must reproduce E032), RTN16,
AT15 (E037's C/AT unchanged), AT16, and L (llm-compressor's codes/scales from lc.py, if E039_LC names its npz).
Calibration: E037's first 128 C4 documents with >=2048 tokens (token-id hash checked against E037).
Env: E039_VARIANTS (default "AT15 AT16"), E039_LC (npz + .json written by lc.py).
"""
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose as D  # noqa: E402  (sets up the E017 import path)
from quantize import tensor_sha  # noqa: E402

I, WINDOW, GROUP, DAMP = D.I, 2048, 128, 0.01
E032 = {'O': 8.605813234340317, 'RTN': 9.734986752455836}
E037_AT = 9.089416695696391  # E037's C/AT test perplexity (reported against AT15, not asserted)
E037_C_IDS = '585ca1c351199d44b90579e7287f7381e76130c5bae61c47ffb9346617674bec'
GRIDS = {15: (lambda a: a / 7, -7, 7), 16: (lambda a: 2 * a / 15, -8, 7)}
STAGES = [['q_proj', 'k_proj', 'v_proj'], ['o_proj'], ['gate_proj', 'up_proj'], ['down_proj']]
F = torch.nn.functional


def rtn(w, grid=15):
    """w [N,K] FP32 -> FP16 reconstruction; grid 15 is the project quantizer (checked against E031's head below)."""
    div, lo, hi = GRIDS[grid]
    g = w.view(w.shape[0], -1, GROUP)
    s = div(g.abs().amax(2, keepdim=True)).half().float()
    return ((torch.where(s > 0, g / s, 0).round().clamp(lo, hi) + 0.0) * s).half().view_as(w)


def gptq(w, h, grid, static=True):
    """E037's gptq() without M: w [N,K] FP32, h [K,K] FP32 -> (reconstruction FP16, codes int8, scales FP16 [N,K/128])."""
    div, lo, hi = GRIDS[grid]
    w, (n, k) = w.clone(), w.shape
    h = h.double()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    scales = torch.empty((n, k // GROUP), dtype=torch.float16, device=w.device)
    if static:  # scales belong to the stored (unpermuted) groups
        scales = div(w.view(n, -1, GROUP).abs().amax(2)).half()
        perm = h.diagonal().argsort(descending=True)
        w, h = w[:, perm], h[perm][:, perm]
        cols = scales.float().repeat_interleave(GROUP, 1)[:, perm]
    h.diagonal().add_(DAMP * h.diagonal().mean())
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    assert torch.isfinite(u).all()
    del h
    codes = torch.empty((n, k), dtype=torch.int8, device=w.device)
    for i in range(k):
        if static:
            s = cols[:, i]
        elif i % GROUP == 0:
            scales[:, i // GROUP] = div(w[:, i:i + GROUP].abs().amax(1)).half()
            s = scales[:, i // GROUP].float()
        q = torch.where(s > 0, w[:, i] / s, 0).round().clamp(lo, hi)
        codes[:, i] = q
        err = (w[:, i] - (q * s).half().float()) / u[i, i]
        if i + 1 < k:
            w[:, i + 1:].addr_(err, u[i, i + 1:], alpha=-1)
    del w, u
    if static:
        codes = codes[:, perm.argsort()]
    return decode(codes, scales, grid), codes, scales


def decode(codes, scales, grid):
    _, lo, hi = GRIDS[grid]
    rec = (codes.float() * scales.float().repeat_interleave(GROUP, 1)).half()
    assert torch.isfinite(rec).all() and lo <= int(codes.min()) and int(codes.max()) <= hi
    return rec


def record(w, h, rec):
    """Layer-output error tr(dW H dW^T) relative to 15-grid plain rounding, and relative weight errors."""
    def one(r):
        d = r.float() - w
        return float(((d @ h) * d).sum()), float(d.norm() / w.norm())
    (pr, wr), (pg, wg) = one(rtn(w)), one(rec)
    return dict(proxy_ratio=pg / pr if pr else None, weight_rel_l2_rtn=wr, weight_rel_l2=wg)


class Stop(Exception):
    pass


with torch.no_grad():
    out, (test, valid, c4), head_hash, ncal = Path(sys.argv[1]), sys.argv[2:5], sys.argv[5], int(sys.argv[6])
    variants = os.environ.get('E039_VARIANTS', 'AT15 AT16').split()
    assert set(variants) <= {'AT15', 'AT16'}
    lc = os.environ.get('E039_LC')
    full = len(sys.argv) <= 7
    work = I.Workload(D.E017 / 'results/e017_20261109', 'O')
    work.model.requires_grad_(False)
    body = work.model.model
    tok = lambda text: work.tokenizer(text, add_special_tokens=False)['input_ids']
    raw = {name: Path(p).read_bytes() for name, p in (('test', test), ('valid', valid))}
    ids = {s: tok(raw[s].decode()) for s in ('valid', 'test')}
    nwin = {s: len(v) // WINDOW if full else int(sys.argv[7]) for s, v in ids.items()}

    rows, seen = [], 0
    with gzip.open(c4, 'rt') as f:  # E037: the first NCAL documents with at least 2048 tokens, their first 2048 tokens
        for line in f:
            seen += 1
            x = tok(json.loads(line)['text'])
            if len(x) >= WINDOW:
                rows.append(x[:WINDOW])
                if len(rows) == ncal:
                    break
    cal = torch.tensor(rows)
    cal_sha = tensor_sha(cal.numpy())
    if ncal == 128:
        assert cal_sha == E037_C_IDS, 'not E037 calibration windows'

    fp16_head = work.model.lm_head.weight.detach().clone()
    assert tensor_sha(np.ascontiguousarray(rtn(fp16_head.float()).T.cpu().numpy())) == head_hash, 'rtn() is not the project quantizer'

    allmods = {}
    for b, layer in enumerate(body.layers):
        allmods[b] = {name: m for name, m in layer.named_modules() if isinstance(m, I.Linear)}
        assert len(allmods[b]) == 7
    orig = {(b, n): m.weight for b, mods in allmods.items() for n, m in mods.items()}
    hf = {(b, n): f'model.layers.{b}.{n}' for b, n in orig}

    def setw(m, rec):
        m.weight = torch.nn.Parameter(rec, requires_grad=False)

    def restore():
        for (b, n), p in orig.items():
            allmods[b][n].weight = p

    arms = {}

    def score(tag):
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
        (out / 'arms.json').write_text(json.dumps(arms, indent=1))
        print('ARM', tag, {s: round(arms[tag][s]['perplexity'], 4) for s in ids}, flush=True)

    score('O')
    for grid, tag in ((15, 'RTN'), (16, 'RTN16')):
        for (b, n), p in orig.items():
            setw(allmods[b][n], rtn(p.detach().float(), grid))
        score(tag)
    restore()

    lc_info = None
    if lc:  # llm-compressor's codes/scales: same original weights, conversion checked in lc.py, decoded here on grid 16
        lc_info = json.loads(Path(lc).with_suffix('.json').read_text())
        assert lc_info['calibration_ids_sha256'] == cal_sha, 'llm-compressor saw different calibration tokens'
        data = np.load(lc)
        for key, p in orig.items():
            assert lc_info['original_sha256'][hf[key]] == tensor_sha(p.detach().cpu().numpy()), ('different original weights', hf[key])
            setw(allmods[key[0]][key[1]], decode(torch.from_numpy(data[hf[key] + '.codes']).cuda(),
                                                 torch.from_numpy(data[hf[key] + '.scales']).cuda(), 16))
        score('L')
        restore()

    # layer-0 inputs of the calibration set, and the replay check (E037)
    box = {}

    def grab(module, args, kwargs):
        box.update(args=args, kwargs=kwargs)
        raise Stop

    handle = body.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    got = []
    for j in range(ncal):
        try:
            body(input_ids=cal[j:j + 1].cuda(), use_cache=False)
        except Stop:
            pass
        assert len(box['args']) == 1
        got.append(box['args'][0][0].cpu())
    handle.remove()
    inps0, kwargs = torch.stack(got), box['kwargs']
    step = lambda layer, x: (lambda y: y[0] if isinstance(y, tuple) else y)(layer(x, **kwargs))
    x = inps0[:1].cuda()
    for layer in body.layers:
        x = step(layer, x)
    assert torch.equal(body.norm(x), body(input_ids=cal[:1].cuda(), use_cache=False).last_hidden_state), 'layer replay differs'
    del x

    layers_log, codes_used = {}, {}
    for tag in variants:
        grid = int(tag[2:])
        restore()
        cur = inps0.cuda()
        outs, low = torch.empty_like(cur), 0
        for b, layer in enumerate(body.layers):
            mods = allmods[b]
            for stage in STAGES:  # T: staged projections, inputs re-collected per stage
                sel = {n: m for n, m in mods.items() if n.rsplit('.', 1)[-1] in stage}
                hs = {n: torch.zeros((m.k, m.k), device='cuda') for n, m in sel.items()}

                def collect(n):
                    def hook(module, args):
                        x = args[0].reshape(-1, module.k).float()
                        hs[n] += x.T @ x
                    return hook

                handles = [m.register_forward_pre_hook(collect(n)) for n, m in sel.items()]
                for j in range(ncal):
                    step(layer, cur[j:j + 1])
                for h in handles:
                    h.remove()
                for n, m in sel.items():
                    w = m.weight.detach().float()
                    rec, codes, scales = gptq(w, hs[n], grid)  # A: static activation order
                    layers_log[f'{tag}:{b}.{n}'] = record(w, hs[n], rec)
                    low += int((codes == -8).sum())
                    setw(m, rec)
                    del w, rec, codes, scales
                del hs
            print('BLOCK', tag, b, {n: round(layers_log[f'{tag}:{b}.{n}']['proxy_ratio'], 4) for n in mods}, flush=True)
            for j in range(ncal):
                outs[j] = step(layer, cur[j:j + 1])[0]
            cur, outs = outs, cur
        codes_used[tag] = dict(code_minus8=low)
        del cur, outs
        torch.cuda.empty_cache()
        score(tag)

    ppl = lambda a, s='test': arms[a][s]['perplexity']
    if full:
        for a, p in E032.items():
            assert abs(ppl(a) / p - 1) < 1e-5, (a, ppl(a), p)
    label = lambda r: 'negligible' if r <= 1 else 'modest' if r <= 5 else 'substantial'
    rel = lambda a, b, s: 100 * (ppl(a, s) / ppl(b, s) - 1)
    pairs = [(a, 'O') for a in arms if a != 'O'] + [('RTN16', 'RTN'), ('AT16', 'AT15'), ('L', 'AT16')]
    compare = {f'{a} vs {b}': {s: dict(rise_pct=rel(a, b, s), label=label(rel(a, b, s))) for s in ids}
               for a, b in pairs if a in arms and b in arms}
    result = dict(sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}, windows=nwin,
                  scored_tokens={s: n * (WINDOW - 1) for s, n in nwin.items()},
                  calibration=dict(windows=ncal, c4_documents_read=seen, ids_sha256=cal_sha, damp=DAMP),
                  gpu=work.identity['gpu'], variants=variants, e037_c_at_test=E037_AT, codes_used=codes_used,
                  llm_compressor=lc_info and {k: v for k, v in lc_info.items() if k != 'original_sha256'},
                  arms=arms, compare=compare, layers=layers_log)
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(compare), flush=True)
