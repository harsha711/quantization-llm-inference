#!/usr/bin/env python3
"""E027: write the E017 project G128 artifacts as an AutoGPTQ-format checkpoint for vLLM. convert.py OUTDIR

Same codes and FP16 scales, no requantization: the project packing ([K/8,N] words, nibble j = input row 8i+j)
is already AutoGPTQ's qweight layout. Symmetric 4-bit, so vLLM's Marlin uses the fixed offset 8 and ignores qzeros.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

root = Path(os.environ['E017_ROOT']) / 'results/e017_20261109'
src = Path(json.loads((root / 'model.json').read_text())['path'])
out = Path(sys.argv[1])
out.mkdir()
manifest = json.loads((root / 'artifacts/manifest.json').read_text())
state = {}
for shard in sorted(src.glob('*.safetensors')):
    state.update(load_file(shard))
errors = {}
for name, meta in manifest['layers'].items():
    w = state.pop(name + '.weight').float().numpy()  # nn.Linear layout [N,K]
    with np.load(root / 'artifacts' / f'{name}.npz') as f:
        packed, scales = f['packed'], f['scales']
    k, n = meta['K'], meta['N']
    assert w.shape == (n, k) and packed.shape == (k // 8, n) and scales.shape == (k // 128, n)
    # self-check: decoding the checkpoint tensors the AutoGPTQ way must give the quantized original weight
    codes = np.stack([(packed >> (4 * j)) & 15 for j in range(8)], 1).reshape(k, n).astype(np.float32)
    w_hat = (codes - 8) * np.repeat(scales.astype(np.float32), 128, 0)
    errors[name] = float(np.linalg.norm(w_hat - w.T) / np.linalg.norm(w))
    assert errors[name] < 0.5, (name, errors[name])  # a wrong axis or nibble order gives > 1
    state[name + '.qweight'] = torch.from_numpy(packed.view(np.int32).copy())
    state[name + '.scales'] = torch.from_numpy(scales.copy())
    state[name + '.qzeros'] = torch.full((k // 128, n // 8), 0x77777777, dtype=torch.int32)
    state[name + '.g_idx'] = torch.arange(k, dtype=torch.int32) // 128
assert len(errors) == 196
save_file(state, out / 'model.safetensors', metadata={'format': 'pt'})
config = json.loads((src / 'config.json').read_text())
config['quantization_config'] = dict(quant_method='gptq', bits=4, group_size=128, sym=True, desc_act=False,
                                     checkpoint_format='gptq', model_file_base_name='model')
(out / 'config.json').write_text(json.dumps(config, indent=1))
for f in src.iterdir():
    if f.is_file() and f.name != 'config.json' and f.suffix != '.safetensors':
        shutil.copy(f, out / f.name)
(out / 'conversion.json').write_text(json.dumps(dict(source=str(src), quantization_rel_l2=errors), indent=1))
print('CONVERTED', len(errors), min(errors.values()), max(errors.values()))
