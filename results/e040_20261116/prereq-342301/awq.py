#!/usr/bin/env python3
"""E040: llm-compressor AWQ arms on the 196 block projections. awq.py ARM OUT.npz MODEL_DIR C4 NCAL

ARM: AWQs  AWQ smoothing, then QuantizationModifier W4A16 (symmetric, group 128)
     AWQa  AWQ smoothing, then QuantizationModifier W4A16_ASYM (zero points, AWQ's usual setting; not the runtime format)
     AWQG  AWQ smoothing, then GPTQModifier W4A16 (symmetric)
All other settings are llm-compressor 0.15.0 defaults, except offload_device=cpu for AWQ's cached activations (memory only).
AWQ rescales norms and biases as well as weights, so OUT.npz holds every tensor that changed: the 196 reconstructed FP16
weights (compressed-tensors' own fake_quantize of the stored scales/zero points, checked to be 4-bit) and the changed
norm weights / biases. Embedding, output head and final norm must be unchanged.
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import compressed_tensors
import llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

GROUP, WINDOW = 128, 2048


def tensor_sha(array):  # scripts/e017/quantize.py
    digest = hashlib.sha256()
    dtype = array.dtype.newbyteorder('<')
    for start in range(0, array.shape[0], GROUP):
        digest.update(memoryview(np.ascontiguousarray(array[start:start + GROUP], dtype=dtype)).cast('B'))
    return digest.hexdigest()


arm, out, path, c4, ncal = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
quant = dict(targets=['Linear'], ignore=['lm_head'])
recipe = [AWQModifier(offload_device=torch.device('cpu')),
          {'AWQs': lambda: QuantizationModifier(scheme='W4A16', **quant),
           'AWQa': lambda: QuantizationModifier(scheme='W4A16_ASYM', **quant),
           'AWQG': lambda: GPTQModifier(scheme='W4A16', **quant)}[arm]()]
assert torch.cuda.get_device_capability() == (12, 0)
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).eval().to('cuda')
lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith('model.layers.')}
assert len(lins) == 196
before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
original = {n: tensor_sha(m.weight.detach().cpu().numpy()) for n, m in lins.items()}

rows, seen = [], 0
with gzip.open(c4, 'rt') as f:  # E037's calibration windows
    for line in f:
        seen += 1
        x = tok(json.loads(line)['text'], add_special_tokens=False)['input_ids']
        if len(x) >= WINDOW:
            rows.append(x[:WINDOW])
            if len(rows) == ncal:
                break
assert len(rows) == ncal
ds = Dataset.from_dict({'input_ids': rows, 'attention_mask': [[1] * WINDOW] * ncal})
oneshot(model=model, dataset=ds, recipe=recipe, tokenizer=tok, max_seq_length=WINDOW, num_calibration_samples=ncal,
        shuffle_calibration_samples=False)

arrays, stats, changed = {}, {}, []
with torch.no_grad():
    for n, m in lins.items():
        w, s = m.weight.detach(), m.weight_scale.detach()
        nn, k = w.shape
        assert s.shape == (nn, k // GROUP), (n, tuple(s.shape))
        zp = getattr(m, 'weight_zero_point', None)
        args = m.quantization_scheme.weights
        assert args.num_bits == 4 and args.group_size == GROUP and args.strategy == 'group', (n, str(args))
        assert getattr(args, 'actorder', None) in (None, 'static') or str(args.actorder).endswith('STATIC'), (n, str(args.actorder))
        rec = fake_quantize(w, s, zp if zp is not None else torch.zeros_like(s, dtype=torch.int8), args).half()
        sf = s.float().repeat_interleave(GROUP, 1)
        c = (rec.float() / sf).round()  # integer offsets q - zp
        assert torch.equal((c * sf).half(), rec), (n, 'reconstruction is not FP16(c*s) with integer c')
        span = c.view(nn, -1, GROUP)
        assert int((span.amax(2) - span.amin(2)).max()) <= 15, (n, 'more than 16 levels in a group')
        arrays[n + '.weight'] = rec.cpu().numpy()
        stats[n] = dict(already_fake_quantized=bool(torch.equal(rec, w.half())), zero_points_nonzero=int((zp != 0).sum()) if zp is not None else 0,
                        rel_l2_vs_original=float((rec.float() - before[n + '.weight'].cuda().float()).norm() / before[n + '.weight'].float().norm()))
    for n, p in model.named_parameters():
        if n not in before or n.endswith('.weight') and n[:-7] in lins:
            continue
        if not torch.equal(p.detach().cpu(), before[n]):
            assert n.startswith('model.layers.') and (n.endswith('layernorm.weight') or n.endswith('.bias')), ('unexpected change', n)
            arrays[n] = p.detach().half().cpu().numpy()
            changed.append(n)
for n in ('model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'):
    if n in before:
        assert torch.equal(dict(model.named_parameters())[n].detach().cpu(), before[n]), ('changed', n)
np.savez(out, **arrays)
info = dict(arm=arm, versions=dict(llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
                                   torch=torch.__version__, transformers=transformers.__version__),
            recipe=[json.loads(json.dumps(r.model_dump(), default=str)) for r in recipe],
            scheme=str(next(iter(lins.values())).quantization_scheme),
            calibration_ids_sha256=tensor_sha(torch.tensor(rows).numpy()), c4_documents_read=seen, ncal=ncal,
            npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), original_sha256=original, changed_other=changed,
            linears=stats, already_fake_quantized=sum(v['already_fake_quantized'] for v in stats.values()),
            zero_points_nonzero=sum(v['zero_points_nonzero'] for v in stats.values()))
out.with_suffix('.json').write_text(json.dumps(info, indent=1))
print('DONE', json.dumps({k: v for k, v in info.items() if k not in ('original_sha256', 'linears')}), flush=True)
