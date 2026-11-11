#!/usr/bin/env python3
"""E035: add an 8-bit group-128 output head (E034's B8) to the E027 AutoGPTQ checkpoint for vLLM. head8conv.py BASEDIR OUTDIR

Symmetric, scale FP16(max|w|/127) per 128 input rows and output column, q = clip(rint(w/s), -127, 127), stored as q+128,
four per int32 word (byte j = input row 4i+j). The blocks stay 4-bit; `dynamic` overrides the bit width for lm_head only.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

B8_REL_L2 = 0.006722508929669857  # E034's measured weight error of this quantizer on this matrix
base, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir()
state = load_file(base / 'model.safetensors')
assert 'lm_head.weight' not in state
w = state['model.embed_tokens.weight'].to(torch.float16).numpy()  # [N,K]
n, k = w.shape
packed, scales, num, den = np.zeros((k // 4, n), np.uint32), np.empty((k // 128, n), np.float16), 0.0, 0.0
for start in range(0, k, 128):
    block = np.ascontiguousarray(w[:, start:start + 128].T).astype(np.float32)  # [128,N]
    s = (np.abs(block).max(0) / np.float32(127)).astype(np.float16)
    s32 = s.astype(np.float32)
    q = np.clip(np.rint(np.divide(block, s32, out=np.zeros_like(block), where=s32 > 0)), -127, 127)
    scales[start // 128] = s
    u = (q + 128).astype(np.uint32)
    for j in range(4):
        packed[start // 4:(start + 128) // 4] |= u[j::4] << (8 * j)
    # self-check: decoding the stored words gives the codes back
    assert np.array_equal(np.stack([(packed[start // 4:(start + 128) // 4] >> (8 * j)) & 255 for j in range(4)], 1).reshape(128, n), u)
    rec = (q * s32).astype(np.float16).astype(np.float64)
    num += float(np.square(rec - block).sum()); den += float(np.square(block.astype(np.float64)).sum())
error = (num / den) ** 0.5
assert abs(error - B8_REL_L2) < 1e-5, error
state['lm_head.qweight'] = torch.from_numpy(packed.view(np.int32).copy())
state['lm_head.scales'] = torch.from_numpy(scales)
state['lm_head.qzeros'] = torch.from_numpy(np.full((k // 128, n // 4), 0x80808080, np.uint32).view(np.int32).copy())
state['lm_head.g_idx'] = torch.arange(k, dtype=torch.int32) // 128
save_file(state, out / 'model.safetensors', metadata={'format': 'pt'})
config = json.loads((base / 'config.json').read_text())
config['quantization_config'].update(lm_head=True, dynamic={r'+:.*lm_head.*': {'bits': 8}})
config['tie_word_embeddings'] = False  # the embedding table stays FP16
(out / 'config.json').write_text(json.dumps(config, indent=1))
for f in base.iterdir():
    if f.is_file() and f.name not in ('config.json', 'model.safetensors', 'conversion.json'):
        shutil.copy(f, out / f.name)
(out / 'conversion.json').write_text(json.dumps(dict(base=str(base), head_quantization_rel_l2=error), indent=1))
print('HEAD8 CONVERTED', n, k, error)
