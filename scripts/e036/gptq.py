#!/usr/bin/env python3
"""E036: perplexity with a calibrated (GPTQ) quantizer in the project format. gptq.py OUT TEST TRAIN HEADHASH NCAL [WINDOWS]

Same runtime format as scripts/e017/quantize.py: symmetric, groups of 128 along K, integer codes in [-7,7], FP16 scale,
reconstruction FP16(q*s). Only the choice of codes and scales changes: GPTQ (Frantar et al. 2022) quantizes one input column
at a time and compensates the remaining columns through the inverse Hessian of the layer inputs (H = sum x^T x over NCAL
calibration windows of the TRAIN split). Each group's scale is max|w|/7 of the already compensated weights.
Blocks are calibrated in order, each on the outputs of the already quantized blocks before it (all seven projections of a
block from one pass). The head is calibrated on the final hidden states of the blocks it is paired with.

Arms (blocks/head), all PyTorch FP16 on reconstructed weights, E032's windows and metric:
  O/F  original (control: E032 O)        R/F  RTN blocks (control: E032 R)     R/G128  RTN head (control: E032 RH)
  R/GPTQ  RTN blocks, GPTQ head          C/F  GPTQ blocks, FP16 head           C/GPTQ  GPTQ blocks and head
"""
import gc
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

I, WINDOW, GROUP, DAMP = D.I, 2048, 128, 0.01
E032 = {'O/F': 8.605813234340317, 'R/F': 9.734986752455836, 'R/G128': 10.57179109043232}


def rtn(w):
    """w [N,K] FP32 -> FP16 reconstruction, the project quantizer (checked bit for bit against E031's head below)."""
    g = w.view(w.shape[0], -1, GROUP)
    s = (g.abs().amax(2, keepdim=True) / 7).half().float()
    return ((torch.where(s > 0, g / s, 0).round().clamp(-7, 7) + 0.0) * s).half().view_as(w)


def gptq(w, h):
    """w [N,K] FP32, h [K,K] FP32 -> (reconstruction FP16 [N,K], codes int8 [N,K], scales FP16 [N,K/128])."""
    w, (n, k) = w.clone(), w.shape
    h = h.double()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(DAMP * h.diagonal().mean())
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True).float()
    assert torch.isfinite(u).all()
    del h
    codes = torch.empty((n, k), dtype=torch.int8, device=w.device)
    scales = torch.empty((n, k // GROUP), dtype=torch.float16, device=w.device)
    for i in range(k):
        if i % GROUP == 0:
            scales[:, i // GROUP] = (w[:, i:i + GROUP].abs().amax(1) / 7).half()
            s = scales[:, i // GROUP].float()
        q = torch.where(s > 0, w[:, i] / s, 0).round().clamp(-7, 7)
        codes[:, i] = q
        err = (w[:, i] - (q * s).half().float()) / u[i, i]
        if i + 1 < k:
            w[:, i + 1:].addr_(err, u[i, i + 1:], alpha=-1)
    del w, u
    rec = (codes.float() * scales.float().repeat_interleave(GROUP, 1)).half()  # from integer codes: in the runtime format by construction
    assert torch.isfinite(rec).all() and int(codes.abs().max()) <= 7
    return rec, codes, scales


def record(w, h, rec):
    """Layer-output error tr(dW H dW^T) on the calibration inputs for RTN and GPTQ, and relative weight error."""
    def one(r):
        d = r.float() - w
        return float(((d @ h) * d).sum()), float(d.norm() / w.norm())
    (pr, wr), (pg, wg) = one(rtn(w)), one(rec)
    assert pg <= pr, ('GPTQ did not reduce the calibration output error', pg, pr)
    return dict(proxy_rtn=pr, proxy_gptq=pg, weight_rel_l2_rtn=wr, weight_rel_l2_gptq=wg,
                reconstructed_sha256=tensor_sha(rec.cpu().numpy()))


def evaluate(work, ids, windows, heads, prefix, out):
    for i in range(windows):
        x = torch.tensor(ids[i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
        hidden = work.model.model(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
        for name, w in heads.items():
            logits = torch.nn.functional.linear(hidden, w).float()
            assert torch.isfinite(logits).all()
            r = out.setdefault(f'{prefix}/{name}', dict(nll=0.0, correct=0, top=[]))
            r['nll'] += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()
            r['top'].append(logits.argmax(-1).cpu().numpy().astype(np.int32))
            r['correct'] += int((logits.argmax(-1) == x[0, 1:]).sum())


class Stop(Exception):
    pass


with torch.inference_mode():
    out, test, train, head_hash, ncal = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    arms, layers_log = {}, {}

    # --- RTN blocks (R): controls and the GPTQ head calibrated on R's hidden states
    work = I.Workload(D.E017 / 'results/e017_20261109', 'R')
    raw, raw_train = test.read_bytes(), train.read_bytes()
    ids = work.tokenizer(raw.decode(), add_special_tokens=False)['input_ids']
    cal_ids = work.tokenizer(raw_train.decode(), add_special_tokens=False)['input_ids']
    windows = int(sys.argv[6]) if len(sys.argv) > 6 else len(ids) // WINDOW
    stride = len(cal_ids) // ncal
    assert stride >= WINDOW
    cal = torch.tensor([cal_ids[j * stride:j * stride + WINDOW] for j in range(ncal)], device='cuda')
    fp16_head = work.model.lm_head.weight.detach().clone()  # [N,K]
    head = fp16_head.float()
    g128 = rtn(head)
    assert tensor_sha(np.ascontiguousarray(g128.T.cpu().numpy())) == head_hash, 'rtn() is not the project quantizer / not the head E031 timed'
    hh = torch.zeros((head.shape[1],) * 2, device='cuda')
    for j in range(ncal):
        x = work.model.model(input_ids=cal[j:j + 1], use_cache=False).last_hidden_state[0].float()
        hh += x.T @ x
    r_head = gptq(head, hh)[0]
    layers_log['lm_head@R'] = record(head, hh, r_head)
    print('HEAD@R', layers_log['lm_head@R'], flush=True)
    evaluate(work, ids, windows, dict(F=fp16_head, G128=g128, GPTQ=r_head), 'R', arms)
    del work, g128, r_head, hh
    gc.collect()
    torch.cuda.empty_cache()

    # --- original model (O): control, then GPTQ of the 28 blocks in order
    work = I.Workload(D.E017 / 'results/e017_20261109', 'O')
    assert torch.equal(work.model.lm_head.weight, fp16_head)
    evaluate(work, ids, windows, dict(F=fp16_head), 'O', arms)
    body = work.model.model
    box = {}

    def grab(module, args, kwargs):
        box.update(args=args, kwargs=kwargs)
        raise Stop

    handle = body.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    inps = []
    for j in range(ncal):
        try:
            body(input_ids=cal[j:j + 1], use_cache=False)
        except Stop:
            pass
        assert len(box['args']) == 1, 'unexpected decoder layer call'
        inps.append(box['args'][0][0].clone())
    handle.remove()
    inps, kwargs = torch.stack(inps), box['kwargs']  # identical shapes for every sample, so one set of kwargs serves all
    step = lambda layer, x: (lambda y: y[0] if isinstance(y, tuple) else y)(layer(x, **kwargs))
    x = inps[:1]
    for layer in body.layers:  # the layer-by-layer replay used for calibration must be the model's own forward
        x = step(layer, x)
    assert torch.equal(body.norm(x), body(input_ids=cal[:1], use_cache=False).last_hidden_state), 'layer replay differs from the model'
    outs, total = torch.empty_like(inps), 0
    for b, layer in enumerate(body.layers):
        mods = {name: m for name, m in layer.named_modules() if isinstance(m, I.Linear)}
        assert len(mods) == 7
        hs = {name: torch.zeros((m.k, m.k), device='cuda') for name, m in mods.items()}

        def collect(name):
            def hook(module, args):
                x = args[0].reshape(-1, module.k).float()
                hs[name] += x.T @ x
            return hook

        handles = [m.register_forward_pre_hook(collect(name)) for name, m in mods.items()]
        for j in range(ncal):
            step(layer, inps[j:j + 1])
        for h in handles:
            h.remove()
        for name, m in mods.items():
            w = m.weight.detach().float()
            rec = gptq(w, hs[name])[0]
            layers_log[f'{b}.{name}'] = record(w, hs[name], rec)
            m.weight = torch.nn.Parameter(rec, requires_grad=False)
            total += 1
            del w, rec
        del hs
        for j in range(ncal):
            outs[j] = step(layer, inps[j:j + 1])[0]
        inps, outs = outs, inps
        print('BLOCK', b, {n: round(layers_log[f'{b}.{n}']['proxy_gptq'] / layers_log[f'{b}.{n}']['proxy_rtn'], 4) for n in mods}, flush=True)
    assert total == 196
    hh = torch.zeros((head.shape[1],) * 2, device='cuda')
    for j in range(ncal):
        x = body.norm(inps[j:j + 1])[0].float()
        hh += x.T @ x
    del inps, outs
    c_head = gptq(head, hh)[0]
    layers_log['lm_head@C'] = record(head, hh, c_head)
    print('HEAD@C', layers_log['lm_head@C'], flush=True)
    del hh, head
    evaluate(work, ids, windows, dict(F=fp16_head, GPTQ=c_head), 'C', arms)

    scored = windows * (WINDOW - 1)
    ppl = {a: math.exp(r['nll'] / scored) for a, r in arms.items()}
    assert all(1 < p < 1e4 for p in ppl.values())
    if windows == len(ids) // WINDOW:  # full run: the three controls must reproduce E032
        for a, p in E032.items():
            assert abs(ppl[a] / p - 1) < 1e-5, (a, ppl[a], p)
    rise = lambda a, b: 100 * (ppl[a] / ppl[b] - 1)
    label = lambda r: 'negligible' if r <= 1 else 'modest' if r <= 5 else 'substantial'
    compare = {f'{a} vs {b}': dict(rise_pct=rise(a, b), label=label(rise(a, b))) for a, b in [
        ('R/F', 'O/F'), ('C/F', 'O/F'), ('R/G128', 'R/F'), ('R/GPTQ', 'R/F'), ('C/GPTQ', 'C/F'), ('R/G128', 'O/F'), ('C/GPTQ', 'O/F')]}
    result = dict(test_sha256=hashlib.sha256(raw).hexdigest(), train_sha256=hashlib.sha256(raw_train).hexdigest(), windows=windows,
                  scored_tokens=scored, calibration=dict(windows=ncal, tokens=ncal * WINDOW, stride=stride, damp=DAMP), gpu=work.identity['gpu'],
                  arms={a: dict(nll_sum=r['nll'], perplexity=ppl[a], top1_accuracy_pct=100 * r['correct'] / scored) for a, r in arms.items()},
                  compare=compare, layers=layers_log)
    np.savez_compressed(out / 'top1.npz', **{a.replace('/', '_'): np.concatenate(r['top']) for a, r in arms.items()})
    (out / 'quality.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(dict(arms=result['arms'], compare=compare)), flush=True)
