#!/usr/bin/env python3
"""E053 part a: the multi-arch rebuilds give outputs bitwise equal to the original sm_120 libraries on this GPU.

eqcheck.py OUT OLD_TC OLD_TC_SHA OLD_MARLIN OLD_MARLIN_SHA
New libraries come from the environment (E052_LIB/E053_TC_SHA, E017_LOCAL/E053_MARLIN_SHA). Seven real projection shapes,
M in {1, 2, 16, 64, 65, 128, 2048}, same random input for old and new, every output bit compared.
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t  # noqa: E402

I = t.I
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
old_tc, old_tc_sha, old_m, old_m_sha = sys.argv[2:6]
results = Path(I.__file__).resolve().parents[2] / 'results/e017_20261109'
manifest = json.loads((results / 'artifacts/manifest.json').read_text())

ops = {('tc', 'new'): t.Operator(), ('tc', 'old'): t.Operator(path=old_tc, sha=old_tc_sha)}
new_m, new_m_sha = os.environ['E017_LOCAL'], os.environ['E053_MARLIN_SHA']
for label, path, sha in (('new', new_m, new_m_sha), ('old', old_m, old_m_sha)):
    os.environ['E017_LOCAL'], I.BODY = path, sha
    ops[('marlin', label)] = I.Operator()
os.environ['E017_LOCAL'], I.BODY = new_m, new_m_sha

rows, g = [], torch.Generator(device='cuda').manual_seed(530)
with torch.inference_mode():
    for leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        name = next(r['name'] for r in manifest['coverage'] if r['name'].endswith(leaf))
        data = I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name])
        layers = dict(tc=t.Linear(name, None, data, 'Q', ops[('tc', 'new')]), marlin=I.Linear(name, None, data, 'Q', ops[('marlin', 'new')]))
        for m in (1, 2, 16, 64, 65, 128, 2048):
            a = (torch.randn((m, data['K']), generator=g, device='cuda') * .5).half()
            for kernel, lin in layers.items():
                outs = {}
                for label in ('new', 'old'):
                    c = torch.full((m, data['N']), float('nan'), dtype=torch.float16, device='cuda')
                    ops[(kernel, label)].call(a, lin.qweight, lin.scales, c)
                    outs[label] = c
                torch.cuda.synchronize()
                row = dict(name=name, kernel=kernel, M=m, K=data['K'], N=data['N'],
                           bitwise=bool(torch.equal(outs['new'].view(torch.int16), outs['old'].view(torch.int16))),
                           finite=bool(torch.isfinite(outs['new']).all()))
                rows.append(row)
                print(json.dumps(row), flush=True)
summary = dict(gpu=torch.cuda.get_device_name(0), cases=len(rows), all_bitwise=all(r['bitwise'] and r['finite'] for r in rows), rows=rows)
(out / 'eqcheck.json').write_text(json.dumps(summary, indent=1))
print('EQCHECK', summary['cases'], summary['all_bitwise'], flush=True)
assert summary['all_bitwise']
