#!/usr/bin/env python3
"""E037: which calibrated quantizer in the project 4-bit format costs least? calib.py OUT TEST VALID TRAIN C4 HEADHASH NCAL [WINDOWS]

Runtime format unchanged (scripts/e017/quantize.py): symmetric, groups of 128 along K, integer codes in [-7,7], FP16 scale,
reconstruction FP16(q*s). Variants are SOURCE/KIND. SOURCE of the calibration text: W WikiText-2 train (in-domain, as E036),
C the first C4 shard (out-of-domain). KIND:
  G    E036's GPTQ: natural column order, each group's scale max|w|/7 of the already compensated weights
  letters A, M, T modify G and combine (e.g. AMT):
  A    columns in descending Hessian-diagonal order, scales fixed beforehand per stored group ("static groups")
  M    each scale chosen from 21 shrink factors 1.0..0.5 to minimize sum |w - w_hat|^2.4
  T    the block's projections quantized in four stages (q/k/v, o, gate/up, down), inputs re-collected per stage
  +    the combination of those of A, M, T that alone lowered VALIDATION perplexity below C/G (skipped if already run)
  S    signed-gradient rounding (after AutoRound, Cheng et al. 2023): per block, 200 sign-SGD steps on a rounding offset
       per weight and a shrink factor per scale, minimizing the per-token relative error of the block's output against
       the original block's; FP32 (the first attempt in FP16 with a plain squared error diverged, job 335222)
Blocks are calibrated in order on the outputs of the already quantized blocks. Every variant is scored with the FP16 head on
the WikiText-2 validation and test splits (E032's windows and metric). The C/ variant with the lowest VALIDATION perplexity
is selected; its head is then GPTQ-quantized (E036's recipe, kind G) on its own final hidden states and both are saved.
Env: E037_VARIANTS (space separated), E037_STEPS (sign-SGD steps per block).
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

I, WINDOW, GROUP, DAMP, BATCH = D.I, 2048, 128, 0.01, 4
E032 = {'O': 8.605813234340317, 'RTN': 9.734986752455836}
E036_CF = 9.188  # E036's C/F test perplexity, the reference for W/G (reported; GPU linear algebra is not bit-reproducible)
STAGES = [['q_proj', 'k_proj', 'v_proj'], ['o_proj'], ['gate_proj', 'up_proj'], ['down_proj']]
F = torch.nn.functional


def rtn(w):
    """w [N,K] FP32 -> FP16 reconstruction, the project quantizer (checked bit for bit against E031's head below)."""
    g = w.view(w.shape[0], -1, GROUP)
    s = (g.abs().amax(2, keepdim=True) / 7).half().float()
    return ((torch.where(s > 0, g / s, 0).round().clamp(-7, 7) + 0.0) * s).half().view_as(w)


def scale_of(g, mse):
    """g [N,G,128] FP32 -> FP16 scales [N,G]: max|w|/7, or the best of 21 shrink factors by |error|^2.4."""
    s0 = g.abs().amax(2) / 7
    best = s0.half()
    if mse:
        err = torch.full_like(s0, float('inf'))
        for c in torch.linspace(1, 0.5, 21).tolist():
            s = (c * s0).half()
            sf = s.float()[..., None]
            e = (g - (torch.where(sf > 0, g / sf, 0).round().clamp(-7, 7) * sf).half().float()).abs().pow(2.4).sum(2)
            better = e < err
            best, err = torch.where(better, s, best), torch.where(better, e, err)
    return best


def gptq(w, h, static=False, mse=False):
    """w [N,K] FP32, h [K,K] FP32 -> (reconstruction FP16 [N,K], codes int8 [N,K], scales FP16 [N,K/128])."""
    w, (n, k) = w.clone(), w.shape
    h = h.double()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    scales = torch.empty((n, k // GROUP), dtype=torch.float16, device=w.device)
    if static:  # scales belong to the stored (unpermuted) groups, so the runtime format needs no permutation
        scales = scale_of(w.view(n, -1, GROUP), mse)
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
            scales[:, i // GROUP] = scale_of(w[:, None, i:i + GROUP], mse)[:, 0]
            s = scales[:, i // GROUP].float()
        q = torch.where(s > 0, w[:, i] / s, 0).round().clamp(-7, 7)
        codes[:, i] = q
        err = (w[:, i] - (q * s).half().float()) / u[i, i]
        if i + 1 < k:
            w[:, i + 1:].addr_(err, u[i, i + 1:], alpha=-1)
    del w, u
    if static:
        codes = codes[:, perm.argsort()]
    return decode(codes, scales), codes, scales


def decode(codes, scales):
    """From integer codes: in the runtime format by construction."""
    rec = (codes.float() * scales.float().repeat_interleave(GROUP, 1)).half()
    assert torch.isfinite(rec).all() and int(codes.abs().max()) <= 7
    return rec


def record(w, h, rec):
    """Layer-output error tr(dW H dW^T) on the calibration inputs relative to plain rounding, and relative weight errors."""
    def one(r):
        d = r.float() - w
        return float(((d @ h) * d).sum()), float(d.norm() / w.norm())
    (pr, wr), (pg, wg) = one(rtn(w)), one(rec)
    return dict(proxy_ratio=pg / pr if pr else None, weight_rel_l2_rtn=wr, weight_rel_l2=wg)


def signround(layer, mods, x, tgt, gen, steps):
    """Sign-SGD on rounding offsets V in [-.5,.5] and scale shrink factors A in [.5,1]; loss: block output error on BATCH samples."""
    names = list(mods)
    w = {n: mods[n].weight.detach().float() for n in names}
    smax = {n: w[n].view(w[n].shape[0], -1, GROUP).abs().amax(2) / 7 for n in names}
    V = {n: torch.zeros_like(w[n], requires_grad=True) for n in names}
    A = {n: torch.ones_like(smax[n], requires_grad=True) for n in names}
    params = list(V.values()) + list(A.values())

    def soft(n):  # straight-through rounding
        s = (A[n] * smax[n]).repeat_interleave(GROUP, 1).clamp_min(1e-12)
        z = w[n] / s + V[n]
        return (z + (z.round().clamp(-7, 7) - z).detach()) * s

    losses, bad = [], 0
    layer.float()  # exact both ways for FP16 values; the block's other parameters are not changed
    for t in range(steps):
        idx = torch.randint(len(x), (BATCH,), generator=gen)
        with torch.enable_grad():
            y = torch.func.functional_call(layer, {f'{n}.weight': soft(n) for n in names}, (x[idx].float(),), kwargs32)
            y, t32 = y[0] if isinstance(y, tuple) else y, tgt[idx].float()
            # per-token relative error: a few positions carry activations ~1000x the rest and would own a plain squared error
            loss = ((y - t32).pow(2).sum(-1) / t32.pow(2).sum(-1).clamp_min(1e-6)).mean()
            grads = torch.autograd.grad(loss, params)
        lr = (1 - t / steps) / steps
        for p, g in zip(params, grads):
            bad += int((~torch.isfinite(g)).sum())
            p.sub_(lr * torch.nan_to_num(g, 0.0, 0.0, 0.0).sign())
        for n in names:
            V[n].clamp_(-0.5, 0.5)
            A[n].clamp_(0.5, 1)
        losses.append(float(loss))
    layer.half()
    assert all(math.isfinite(v) for v in losses)
    res = {}
    for n in names:
        scales = (A[n] * smax[n]).detach().half()
        s = scales.float().repeat_interleave(GROUP, 1)
        codes = torch.where(s > 0, w[n] / s + V[n].detach(), 0).round().clamp(-7, 7).to(torch.int8)
        res[n] = (decode(codes, scales), codes, scales)
    k = max(1, steps // 20)
    return res, dict(loss_first=sum(losses[:k]) / k, loss_last=sum(losses[-k:]) / k, nonfinite_grad_elements=bad,
                     offsets_at_bound_pct=100 * sum(float((v.abs() == 0.5).sum()) for v in V.values()) / sum(v.numel() for v in V.values()))


class Stop(Exception):
    pass


with torch.no_grad():
    out, (test, valid, train, c4), head_hash, ncal = Path(sys.argv[1]), sys.argv[2:6], sys.argv[6], int(sys.argv[7])
    variants = os.environ.get('E037_VARIANTS', 'W/G C/G C/A C/M C/T C/+ C/S').split()
    steps = int(os.environ.get('E037_STEPS', 200))
    full = len(sys.argv) <= 8
    work = I.Workload(D.E017 / 'results/e017_20261109', 'O')
    work.model.requires_grad_(False)
    body = work.model.model
    tok = lambda text: work.tokenizer(text, add_special_tokens=False)['input_ids']
    raw = {name: Path(p).read_bytes() for name, p in (('test', test), ('valid', valid), ('train', train))}
    ids = {s: tok(raw[s].decode()) for s in ('valid', 'test')}
    nwin = {s: len(v) // WINDOW if full else int(sys.argv[8]) for s, v in ids.items()}

    t = tok(raw['train'].decode())
    stride = len(t) // ncal
    assert stride >= WINDOW
    cal, rows, seen = {'W': torch.tensor([t[j * stride:j * stride + WINDOW] for j in range(ncal)])}, [], 0
    with gzip.open(c4, 'rt') as f:  # the first NCAL documents of the shard with at least 2048 tokens, their first 2048 tokens
        for line in f:
            seen += 1
            x = tok(json.loads(line)['text'])
            if len(x) >= WINDOW:
                rows.append(x[:WINDOW])
                if len(rows) == ncal:
                    break
    assert len(rows) == ncal
    cal['C'] = torch.tensor(rows)
    print('C4 documents read', seen, flush=True)

    fp16_head = work.model.lm_head.weight.detach().clone()  # [N,K]
    head = fp16_head.float()
    assert tensor_sha(np.ascontiguousarray(rtn(head).T.cpu().numpy())) == head_hash, 'rtn() is not the project quantizer / not the head E031 timed'

    allmods = {}
    for b, layer in enumerate(body.layers):
        allmods[b] = {name: m for name, m in layer.named_modules() if isinstance(m, I.Linear)}
        assert len(allmods[b]) == 7
    orig = {(b, n): m.weight for b, mods in allmods.items() for n, m in mods.items()}

    def setw(m, rec):
        m.weight = torch.nn.Parameter(rec, requires_grad=False)

    arms = {}

    def score(tag, heads):
        for split in ids:
            r = {h: [0.0, 0] for h in heads}
            for i in range(nwin[split]):
                x = torch.tensor(ids[split][i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
                hidden = body(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
                for h, w in heads.items():
                    logits = F.linear(hidden, w).float()
                    assert torch.isfinite(logits).all()
                    r[h][0] += F.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()
                    r[h][1] += int((logits.argmax(-1) == x[0, 1:]).sum())
            n = nwin[split] * (WINDOW - 1)
            for h, (nll, correct) in r.items():
                arms.setdefault(f'{tag}/{h}', {})[split] = dict(nll_sum=nll, perplexity=math.exp(nll / n), top1_accuracy_pct=100 * correct / n)
        print('ARM', tag, {h: {s: round(arms[f'{tag}/{h}'][s]['perplexity'], 4) for s in ids} for h in heads}, flush=True)

    score('O', dict(F=fp16_head))
    for (b, n), p in orig.items():
        setw(allmods[b][n], rtn(p.detach().float()))
    score('RTN', dict(F=fp16_head))

    # layer-0 inputs of both calibration sets (kept on the CPU), and the replay check
    box, inps0 = {}, {}

    def grab(module, args, kwargs):
        box.update(args=args, kwargs=kwargs)
        raise Stop

    for (b, n), p in orig.items():
        allmods[b][n].weight = p
    handle = body.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    for src, c in cal.items():
        got = []
        for j in range(ncal):
            try:
                body(input_ids=c[j:j + 1].cuda(), use_cache=False)
            except Stop:
                pass
            assert len(box['args']) == 1, 'unexpected decoder layer call'
            got.append(box['args'][0][0].cpu())
        inps0[src] = torch.stack(got)
    handle.remove()
    kwargs = box['kwargs']  # identical shapes for every sample, so one set of kwargs serves all
    f32 = lambda v: v.float() if torch.is_tensor(v) and v.is_floating_point() else type(v)(f32(e) for e in v) if isinstance(v, (tuple, list)) else v
    kwargs32 = {k: f32(v) for k, v in kwargs.items()}
    step = lambda layer, x: (lambda y: y[0] if isinstance(y, tuple) else y)(layer(x, **kwargs))
    x = inps0['C'][:1].cuda()
    for layer in body.layers:  # the layer-by-layer replay used for calibration must be the model's own forward
        x = step(layer, x)
    assert torch.equal(body.norm(x), body(input_ids=cal['C'][:1].cuda(), use_cache=False).last_hidden_state), 'layer replay differs from the model'
    del x

    layers_log, saved, hhs, combo = {}, {}, {}, {}
    for tag in variants:
        src, kind = tag.split('/')
        if kind == '+':
            base = arms[f'{src}/G/F']['valid']['perplexity']
            kind = ''.join(f for f in 'AMT' if arms[f'{src}/{f}/F']['valid']['perplexity'] < base)
            combo[tag] = kind
            print('COMBINATION', tag, repr(kind), flush=True)
            if len(kind) < 2:  # nothing or one letter: that variant is already in the table
                continue
        for (b, n), p in orig.items():
            allmods[b][n].weight = p
        cur = inps0[src].cuda()
        outs, keep, gen = torch.empty_like(cur), {}, torch.Generator().manual_seed(0)
        fp = cur.clone() if kind == 'S' else None
        for b, layer in enumerate(body.layers):
            mods = allmods[b]
            if kind == 'S':
                tgt = torch.empty_like(fp)
                for j in range(ncal):
                    tgt[j] = step(layer, fp[j:j + 1])[0]
                res, info = signround(layer, mods, cur, tgt, gen, steps)
                fp = tgt
                del tgt
                for n, (rec, codes, scales) in res.items():
                    setw(mods[n], rec)
                    keep[mods[n].name] = (codes.cpu().numpy(), scales.cpu().numpy())
                layers_log[f'{tag}:{b}'] = info
                print('BLOCK', tag, b, info, flush=True)
            else:
                for stage in (STAGES if 'T' in kind else [sum(STAGES, [])]):
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
                        rec, codes, scales = gptq(w, hs[n], static='A' in kind, mse='M' in kind)
                        layers_log[f'{tag}:{b}.{n}'] = record(w, hs[n], rec)
                        setw(m, rec)
                        keep[m.name] = (codes.cpu().numpy(), scales.cpu().numpy())
                        del w, rec, codes, scales
                    del hs
                print('BLOCK', tag, b, {n: round(layers_log[f'{tag}:{b}.{n}']['proxy_ratio'], 4) for n in mods}, flush=True)
            for j in range(ncal):
                outs[j] = step(layer, cur[j:j + 1])[0]
            cur, outs = outs, cur
        assert len(keep) == 196
        hh = torch.zeros((head.shape[1],) * 2, device='cuda')
        for j in range(ncal):
            x = body.norm(cur[j:j + 1])[0].float()
            hh += x.T @ x
        hhs[tag], saved[tag] = hh.cpu(), keep
        del cur, outs, fp, hh
        torch.cuda.empty_cache()
        score(tag, dict(F=fp16_head))

    # selection on the validation split among the out-of-domain variants, then the head on the selected blocks
    variants = [t for t in variants if f'{t}/F' in arms]
    best = min((t for t in variants if t[0] == 'C'), key=lambda t: arms[f'{t}/F']['valid']['perplexity'])
    names = {m.name: m for mods in allmods.values() for m in mods.values()}
    for name, (codes, scales) in saved[best].items():
        setw(names[name], decode(torch.from_numpy(codes).cuda(), torch.from_numpy(scales).cuda()))
    hh = hhs[best].cuda()
    h_rec, h_codes, h_scales = gptq(head, hh)
    layers_log['lm_head'] = record(head, hh, h_rec)
    score('SEL', dict(F=fp16_head, GPTQ=h_rec))
    for split in ids:  # the reloaded blocks are the ones scored in the sweep
        assert arms['SEL/F'][split]['nll_sum'] == arms[f'{best}/F'][split]['nll_sum'], 'reloaded checkpoint differs'
    ckpt = {f'{name}.{part}': a for name, pair in saved[best].items() for part, a in zip(('codes', 'scales'), pair)}
    ckpt.update({'lm_head.codes': h_codes.cpu().numpy(), 'lm_head.scales': h_scales.cpu().numpy()})
    np.savez(out / 'checkpoint.npz', **ckpt)  # [N,K] int8 codes in [-7,7] and [N,K/128] FP16 scales per layer
    sha = hashlib.sha256()
    with open(out / 'checkpoint.npz', 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            sha.update(chunk)

    ppl = lambda a, s='test': arms[a][s]['perplexity']
    if full:  # the two controls must reproduce E032
        for a, p in E032.items():
            assert abs(ppl(f'{a}/F') / p - 1) < 1e-5, (a, ppl(f'{a}/F'), p)
    label = lambda r: 'negligible' if r <= 1 else 'modest' if r <= 5 else 'substantial'
    rise = lambda a, b, s: (lambda r: dict(rise_pct=r, label=label(r)))(100 * (ppl(a, s) / ppl(b, s) - 1))
    compare = {f'{a} vs {b}': {s: rise(a, b, s) for s in ids} for a, b in
               [(f'{t}/F', 'O/F') for t in ['RTN'] + variants] + [('SEL/GPTQ', 'SEL/F'), ('SEL/GPTQ', 'O/F')]}
    result = dict(sha256={k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}, windows=nwin, scored_tokens={s: n * (WINDOW - 1) for s, n in nwin.items()},
                  calibration=dict(windows=ncal, tokens=ncal * WINDOW, wikitext_stride=stride, c4_documents_read=seen, damp=DAMP, sign_steps=steps, sign_batch=BATCH,
                                   ids_sha256={k: tensor_sha(v.numpy()) for k, v in cal.items()}),
                  gpu=work.identity['gpu'], variants=variants, combination=combo, selected=best, checkpoint_sha256=sha.hexdigest(), e036_c_f_test=E036_CF,
                  arms=arms, compare=compare, layers=layers_log)
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(dict(selected=best, compare=compare)), flush=True)
