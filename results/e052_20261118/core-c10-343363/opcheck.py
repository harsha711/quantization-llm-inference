#!/usr/bin/env python3
"""E052 operator prerequisite: tc_ladder (through t.py's Operator) on the model's real codes/scales, same input, FP64 reference.

opcheck.py OUT [--quick]
Every distinct block shape (first layer's artifact of each type) and the 4-bit head shape (151936 x 1536, contract quantizer on
the model's tied FP16 embedding) at the M values the model uses; all output elements checked under the unchanged GEMM bound.
Also: graph replay and repeat bitwise equal to eager, and a misaligned activation is refused (no silent fallback).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t  # noqa: E402  (patches the E017 bridge to tc_ladder)

I = t.I
out, quick = Path(sys.argv[1]), '--quick' in sys.argv
out.mkdir(parents=True, exist_ok=False)
results = Path(I.__file__).resolve().parents[2] / 'results/e017_20261109'
manifest = json.loads((results / 'artifacts/manifest.json').read_text())
op = t.Operator()
rows, g = [], torch.Generator(device='cuda').manual_seed(52)


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
        row = dict(name=name, M=m, K=k, N=n, kernel='tc_m32' if m >= 64 else 'tc', checked=ref.numel(), bad=int(bad.sum()),
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
        rows.append(row)
        print(json.dumps(row), flush=True)
        del a, c, c2, c3, ref, err, bad, graph
        torch.cuda.empty_cache()


with torch.inference_mode():
    ms = [1, 64, 128] if quick else [1, 2, 16, 63, 64, 65, 128, 2048, 16384]
    for leaf in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        name = next(r['name'] for r in manifest['coverage'] if r['name'].endswith(leaf))
        data = I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name])
        lin = t.Linear(name, None, data, 'Q', op)
        codes = I.e004.unpack_project(data['packed'], data['K'], data['N'])
        w = ((codes.astype(np.float32) - 8) * data['scales'].repeat(128, axis=0).astype(np.float32)).astype(np.float16)
        assert I.array_sha(w) == data['hashes']['reconstructed']
        check(name, lin.qweight, lin.scales, torch.from_numpy(w).cuda(), ms)
    # head: contract quantizer on the tied FP16 embedding, as E031 (no Marlin repack needed for tc_ladder)
    from safetensors import safe_open
    model = json.loads((results / 'model.json').read_text())['path']
    path = next(Path(model).glob('*.safetensors'))
    with safe_open(str(path), 'pt') as f:
        emb = f.get_tensor('model.embed_tokens.weight').half().numpy()  # [N, K]
    data = I.quantize(np.ascontiguousarray(emb.T))
    lin = t.Linear('lm_head', None, dict(data, marlin_packed=None, marlin_scales=None, K=emb.shape[1], N=emb.shape[0]), 'Q', op)
    check('lm_head', lin.qweight, lin.scales, torch.from_numpy(data['reconstructed']).cuda(), [1] if quick else [1, 64])
    # no silent fallback: a 16-byte-misaligned activation must be refused before launch
    a = torch.zeros(1 * 1536 + 1, dtype=torch.float16, device='cuda')[1:].view(1, 1536)
    try:
        op.call(a, lin.qweight, lin.scales, torch.empty((1, emb.shape[0]), dtype=torch.float16, device='cuda'))
        refused = False
    except AssertionError:
        refused = True
summary = dict(cases=len(rows), bad=sum(r['bad'] for r in rows), all_repeat_bitwise=all(r['repeat_bitwise'] for r in rows),
               all_graph_bitwise=all(r['graph_bitwise'] for r in rows), misaligned_refused=refused,
               max_rel_l2=max(r['rel_l2'] for r in rows), max_gpu_vs_cpu_fp64_rel=max(r.get('gpu_vs_cpu_fp64_rel', 0) for r in rows),
               shapes=sorted({(r['K'], r['N']) for r in rows}), Ms=sorted({r['M'] for r in rows}), rows=rows)
I.save(out / 'opcheck.json', summary)
print('OPCHECK', json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)
assert summary['bad'] == 0 and summary['all_repeat_bitwise'] and summary['all_graph_bitwise'] and refused
