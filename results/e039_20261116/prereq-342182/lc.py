#!/usr/bin/env python3
"""E039 arm L: llm-compressor W4A16 GPTQ on the 196 block projections. lc.py OUT.npz MODEL_DIR C4 NCAL

Runs in its own overlay environment (llm-compressor installed over the frozen E017 venv). Calibration: E037's first NCAL
C4 documents with >=2048 tokens, first 2048 tokens. Recipe: GPTQModifier(targets="Linear", scheme="W4A16",
ignore=["lm_head"]), every other setting llm-compressor's default. Writes int8 codes [N,K] and FP16 scales [N,K/128] per
layer (HF module names) and OUT.json; the conversion must reproduce each quantized weight bit for bit as FP16(q*s).
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import compressed_tensors
import llmcompressor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

GROUP, WINDOW = 128, 2048


def tensor_sha(array):  # scripts/e017/quantize.py
    digest = hashlib.sha256()
    dtype = array.dtype.newbyteorder('<')
    for start in range(0, array.shape[0], GROUP):
        digest.update(memoryview(np.ascontiguousarray(array[start:start + GROUP], dtype=dtype)).cast('B'))
    return digest.hexdigest()


out, path, c4, ncal = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
assert torch.cuda.get_device_capability() == (12, 0)
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).eval().to('cuda')
lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n.startswith('model.layers.')}
assert len(lins) == 196
original = {n: tensor_sha(m.weight.detach().cpu().numpy()) for n, m in lins.items()}

rows, seen = [], 0
with gzip.open(c4, 'rt') as f:
    for line in f:
        seen += 1
        x = tok(json.loads(line)['text'], add_special_tokens=False)['input_ids']
        if len(x) >= WINDOW:
            rows.append(x[:WINDOW])
            if len(rows) == ncal:
                break
assert len(rows) == ncal
ds = Dataset.from_dict({'input_ids': rows, 'attention_mask': [[1] * WINDOW] * ncal})
recipe = GPTQModifier(targets='Linear', scheme='W4A16', ignore=['lm_head'])
oneshot(model=model, dataset=ds, recipe=recipe, tokenizer=tok, max_seq_length=WINDOW, num_calibration_samples=ncal,
        shuffle_calibration_samples=False)

arrays, stats = {}, {}
for n, m in lins.items():
    w = m.weight.detach()
    nn, k = w.shape
    s = m.weight_scale.detach()
    assert s.shape == (nn, k // GROUP), (n, tuple(s.shape))
    zp = getattr(m, 'weight_zero_point', None)
    assert zp is None or not zp.any(), (n, 'nonzero zero point')
    gi = getattr(m, 'weight_g_idx', None)
    assert gi is None or torch.equal(gi.cpu().long(), torch.arange(k) // GROUP), (n, 'permuted groups')
    sf = s.float().repeat_interleave(GROUP, 1)
    q = torch.where(sf > 0, w.float() / sf, 0).round()
    assert -8 <= int(q.min()) and int(q.max()) <= 7, n
    assert torch.equal((q * sf).half(), w.half()), (n, 'weight is not FP16(q*s) on contiguous groups')  # bit for bit
    arrays[n + '.codes'] = q.to(torch.int8).cpu().numpy()
    arrays[n + '.scales'] = s.half().cpu().numpy()
    stats[n] = dict(code_minus8=int((q == -8).sum()), code_plus7=int((q == 7).sum()))
np.savez(out, **arrays)
info = dict(versions=dict(llmcompressor=llmcompressor.__version__, compressed_tensors=compressed_tensors.__version__,
                          torch=torch.__version__, transformers=transformers.__version__),
            recipe=json.loads(json.dumps(recipe.model_dump(), default=str)),
            scheme=str(getattr(next(iter(lins.values())), 'quantization_scheme', None)),
            calibration_ids_sha256=tensor_sha(torch.tensor(rows).numpy()), c4_documents_read=seen, ncal=ncal,
            npz_sha256=hashlib.sha256(Path(out).read_bytes()).hexdigest(), original_sha256=original, codes=stats,
            code_minus8_total=sum(v['code_minus8'] for v in stats.values()))
Path(out).with_suffix('.json').write_text(json.dumps(info, indent=1))
print('DONE', json.dumps({k: v for k, v in info.items() if k not in ('original_sha256', 'codes')}), flush=True)
