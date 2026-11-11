#!/usr/bin/env python3
"""E031: add a project-G128 4-bit output head to the E027 AutoGPTQ checkpoint for vLLM. headconv.py BASEDIR OUTDIR

The head is the tied embedding matrix cast to FP16 (as both runtimes load it), quantized by the unchanged E017
quantizer. The FP16 embedding table stays; only the output projection becomes 4-bit.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(os.environ['E017_ROOT']) / 'scripts/e017'))
from quantize import quantize  # noqa: E402

base, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir()
state = load_file(base / 'model.safetensors')
assert 'lm_head.weight' not in state
w = state['model.embed_tokens.weight'].to(torch.float16).numpy()  # [N,K]
n, k = w.shape
data = quantize(np.ascontiguousarray(w.T))
# self-check: decoding the stored tensors the AutoGPTQ way gives the quantizer's reconstruction bit for bit
codes = np.stack([(data['packed'] >> (4 * j)) & 15 for j in range(8)], 1).reshape(k, n).astype(np.float32)
w_hat = ((codes - 8) * np.repeat(data['scales'].astype(np.float32), 128, 0)).astype(np.float16)
assert np.array_equal(w_hat.view(np.uint16), data['reconstructed'].view(np.uint16))
error = float(np.linalg.norm(w_hat.astype(np.float32) - w.T.astype(np.float32)) / np.linalg.norm(w.astype(np.float32)))
state['lm_head.qweight'] = torch.from_numpy(data['packed'].view(np.int32).copy())
state['lm_head.scales'] = torch.from_numpy(data['scales'].copy())
state['lm_head.qzeros'] = torch.full((k // 128, n // 8), 0x77777777, dtype=torch.int32)
state['lm_head.g_idx'] = torch.arange(k, dtype=torch.int32) // 128
save_file(state, out / 'model.safetensors', metadata={'format': 'pt'})
config = json.loads((base / 'config.json').read_text())
config['quantization_config']['lm_head'] = True
config['tie_word_embeddings'] = False  # the embedding table stays FP16; the head has its own 4-bit tensors
(out / 'config.json').write_text(json.dumps(config, indent=1))
for f in base.iterdir():
    if f.is_file() and f.name not in ('config.json', 'model.safetensors', 'conversion.json'):
        shutil.copy(f, out / f.name)
(out / 'conversion.json').write_text(json.dumps(dict(base=str(base), head_quantization_rel_l2=error, hashes=data['hashes']), indent=1))
print('HEAD CONVERTED', n, k, error)
