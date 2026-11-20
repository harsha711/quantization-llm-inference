#!/usr/bin/env python3
"""E054 operator prerequisite on one GPU: tc_ladder (t.py's Operator) or corrected Marlin (E017 bridge, --marlin) on the model's
real codes/scales, same input, FP64 reference. E052's opcheck.py with a Marlin mode added.

opcheck.py OUT [--quick] [--marlin]
Every distinct block shape (first layer's artifact of each type) at the M values the model uses; all output elements checked under
the unchanged GEMM bound; repeat and graph replay bitwise equal to eager. tc_ladder only: the 4-bit head shape (contract quantizer
on the tied FP16 embedding) and the misaligned-activation refusal. Marlin: M <= 2048 (its row bound in the bridge), its own
repacked layout from the E017 artifacts, workspace audit after every call.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t  # noqa: E402  (patches nothing by itself; the bridge objects are chosen below)

I = t.I
out, quick, marlin = Path(sys.argv[1]), '--quick' in sys.argv, '--marlin' in sys.argv
out.mkdir(parents=True, exist_ok=False)
results = Path(I.__file__).resolve().parents[2] / 'results/e017_20261109'
manifest = json.loads((results / 'artifacts/manifest.json').read_text())
if marlin:
    I.BODY = os.environ['E054_MARLIN_SHA']
    op, Linear = I.Operator(), I.Linear
else:
    op, Linear = t.Operator(), t.Linear
rows, g = [], torch.Generator(device='cuda').manual_seed(53)
name_of = (lambda m: 'marlin') if marlin else (lambda m: 'tc_m32' if m >= 64 else 'tc')


def check(name, q, s, w, ms):
    k, n = w.shape
    for m in ms:
        a = (torch.randn((m, k), generator=g, device='cuda') * .5).half()
        c = torch.empty((m, n), dtype=torch.float16, device='cuda')
        op.call(a, q, s, c)
        torch.cuda.synchronize()
        ref = a.double() @ w.double()
        err = (c.double() - ref).abs()
        bad = ~torch.isfinite(c) | (err > 1e-3 * ref.abs() + 1e-3 * ref.pow(2).mean().sqrt())
        row = dict(name=name, M=m, K=k, N=n, kernel=name_of(m), checked=ref.numel(), bad=int(bad.sum()),
                   max_abs=float(err.max()), rel_l2=float(err.norm() / ref.norm()))
        if m <= 64:  # contract reference is CPU FP64; cross-check the GPU one
            cpu = a.cpu().double() @ w.cpu().double()
            row['gpu_vs_cpu_fp64_rel'] = float((ref.cpu() - cpu).norm() / cpu.norm())
        c2 = torch.empty_like(c)
        op.call(a, q, s, c2)
        graph, c3 = torch.cuda.CUDAGraph(), torch.empty_like(c)
        with torch.cuda.graph(graph):
            op.call(a, q, s, c3)
        c3.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        row.update(repeat_bitwise=bool(torch.equal(c.view(torch.int16), c2.view(torch.int16))),
                   graph_bitwise=bool(torch.equal(c.view(torch.int16), c3.view(torch.int16))))
        if marlin:
            row['workspace'] = op.audit()
            row['workspace_ok'] = all(row['workspace'].values())
        rows.append(row)
        print(json.dumps(row), flush=True)
        del a, c, c2, c3, ref, err, bad, graph
        torch.cuda.empty_cache()


refused = None
with torch.inference_mode():
    ms = [1, 64, 128] if quick else [1, 2, 16, 63, 64, 65, 128, 2048] + ([] if marlin else [16384])
    for leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        name = next(r['name'] for r in manifest['coverage'] if r['name'].endswith(leaf))
        data = I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name])
        lin = Linear(name, None, data, 'Q', op)
        codes = I.e004.unpack_project(data['packed'], data['K'], data['N'])
        w = ((codes.astype(np.float32) - 8) * data['scales'].repeat(128, axis=0).astype(np.float32)).astype(np.float16)
        assert I.array_sha(w) == data['hashes']['reconstructed']
        check(name, lin.qweight, lin.scales, torch.from_numpy(w).cuda(), ms)
    if not marlin:  # head: contract quantizer on the tied FP16 embedding, as E031/E052
        from safetensors import safe_open
        model = json.loads((results / 'model.json').read_text())['path']
        path = next(Path(model).glob('*.safetensors'))
        with safe_open(str(path), 'pt') as f:
            emb = f.get_tensor('model.embed_tokens.weight').half().numpy()  # [N, K]
        data = I.quantize(np.ascontiguousarray(emb.T))
        lin = t.Linear('lm_head', None, dict(data, marlin_packed=None, marlin_scales=None, K=emb.shape[1], N=emb.shape[0]), 'Q', op)
        check('lm_head', lin.qweight, lin.scales, torch.from_numpy(data['reconstructed']).cuda(), [1] if quick else [1, 64])
        a = torch.zeros(1 * 1536 + 1, dtype=torch.float16, device='cuda')[1:].view(1, 1536)
        try:
            op.call(a, lin.qweight, lin.scales, torch.empty((1, emb.shape[0]), dtype=torch.float16, device='cuda'))
            refused = False
        except AssertionError:
            refused = True
summary = dict(kernel='marlin' if marlin else 'tc_ladder', gpu=torch.cuda.get_device_name(0),
               capability=list(torch.cuda.get_device_capability(0)), cases=len(rows), bad=sum(r['bad'] for r in rows),
               all_repeat_bitwise=all(r['repeat_bitwise'] for r in rows), all_graph_bitwise=all(r['graph_bitwise'] for r in rows),
               all_workspace_ok=all(r.get('workspace_ok', True) for r in rows), misaligned_refused=refused,
               max_rel_l2=max(r['rel_l2'] for r in rows), max_gpu_vs_cpu_fp64_rel=max(r.get('gpu_vs_cpu_fp64_rel', 0) for r in rows),
               shapes=sorted({(r['K'], r['N']) for r in rows}), Ms=sorted({r['M'] for r in rows}), rows=rows)
I.save(out / 'opcheck.json', summary)
print('OPCHECK', json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)
assert summary['bad'] == 0 and summary['all_repeat_bitwise'] and summary['all_graph_bitwise'] and summary['all_workspace_ok']
assert marlin or refused
