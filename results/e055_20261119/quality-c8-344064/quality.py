#!/usr/bin/env python3
"""E055 (E032's quality.py + variant Q and per-window records): quality.py OUT VARIANT TEXT HEADHASH
Run Q through E052's t.py: `t.py --marlin quality.py ...` = corrected Marlin, `t.py quality.py ...` = tc_ladder.

VARIANT: O original FP16 model; R the 196 block projections replaced by their G128 reconstruction (the model every
experiment since E017 ran); OH / RH the same with the output head also replaced by its G128 reconstruction (E031's head).
All arithmetic is PyTorch FP16 on reconstructed weights: this measures the quantized MODEL, not a runtime.
Non-overlapping 2048-token windows over the whole text, no special tokens; perplexity = exp(mean next-token NLL).
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
with torch.inference_mode():
    out, variant, text, head_hash = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
    assert variant in ('O', 'R', 'OH', 'RH', 'Q')
    label = sys.argv[5] if len(sys.argv) > 5 else variant  # QM / QT: which kernel t.py installed
    work = I.Workload(D.E017 / 'results/e017_20261109', variant[0])
    if variant.endswith('H'):
        w = work.model.lm_head.weight.detach().cpu().numpy()  # [N,K] FP16
        data = quantize(np.ascontiguousarray(w.T))
        assert data['hashes']['reconstructed'] == head_hash, 'not the head E031 timed'
        head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=torch.float16, device='cuda')
        head.weight.copy_(torch.from_numpy(np.ascontiguousarray(data['reconstructed'].T)))
        work.model.lm_head = head
    raw = text.read_bytes()
    ids = work.tokenizer(raw.decode(), add_special_tokens=False)['input_ids']
    windows, nll, correct, top, per = len(ids) // WINDOW, 0.0, 0, [], []
    assert windows >= 50
    for i in range(windows):
        x = torch.tensor(ids[i * WINDOW:(i + 1) * WINDOW], device='cuda')[None]
        logits = work.model(input_ids=x, use_cache=False).logits[0, :-1].float()
        assert torch.isfinite(logits).all()
        w_nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction='sum').double().item()  # E032's arithmetic, kept per window
        per.append(w_nll)
        nll += w_nll
        top.append(logits.argmax(-1).cpu().numpy().astype(np.int32))
        correct += int((logits.argmax(-1) == x[0, 1:]).sum())
    scored = windows * (WINDOW - 1)
    result = dict(variant=label, workload_variant=variant, body=getattr(I, 'BODY', None), text_sha256=hashlib.sha256(raw).hexdigest(), tokens=len(ids), windows=windows, scored_tokens=scored,
                  nll_sum=nll, perplexity=math.exp(nll / scored), top1_accuracy=correct / scored, gpu=work.identity['gpu'])
    assert 1 < result['perplexity'] < 1e4
    np.savez_compressed(out / f'top1-{label}.npz', top1=np.concatenate(top), window_nll=np.array(per))
    (out / f'quality-{label}.json').write_text(json.dumps(result, indent=1))
    print('DONE', json.dumps(result), flush=True)
