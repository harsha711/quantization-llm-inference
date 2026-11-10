#!/usr/bin/env python3
"""E018: locate the E017 Q/R logit divergence. Diagnostic only: no timing, no gate change.

One process holds R (reconstructed FP16) and Q (corrected Marlin) for all 196 projections
and switches per request:
  R / Q    the E017 paths, unchanged modules
  F32/F64  same W_hat, projection GEMM accumulated in FP32/FP64 then rounded to FP16
           (Marlin-free rounding controls)
  P        R trajectory; every projection call also runs Q on the IDENTICAL input and
           checks Q and R against FP64 under the unchanged GEMM tolerance
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

E017 = Path(os.environ['E017_ROOT'])
sys.path.insert(0, str(E017 / 'scripts/e017'))
import integration as I  # noqa: E402
from infer import Runner  # noqa: E402

MODE = 'R'
PROBE, HID = [], []


def gemm_stats(got, ref):
    # CONTRACT.md: abs(got-ref) <= 1e-3*abs(ref) + 1e-3*rms(ref), FP64 reference.
    err = (got.double() - ref).abs()
    bad = ~torch.isfinite(got) | (err > 1e-3 * ref.abs() + 1e-3 * ref.pow(2).mean().sqrt())
    return dict(bad=int(bad.sum()), max_abs=float(err.max()), rel_l2=float(err.norm() / ref.norm()))


class Dual(torch.nn.Module):
    def __init__(self, r, q):
        super().__init__()
        self.r, self.q, self.calls = r, q, 0

    def forward(self, x):
        step, self.calls = self.calls, self.calls + 1
        if MODE == 'R':
            return self.r(x)
        if MODE == 'Q':
            return self.q(x)
        flat = x.reshape(-1, self.r.k).contiguous()
        if MODE in ('F32', 'F64'):
            t = torch.float32 if MODE == 'F32' else torch.float64
            out = (flat.to(t) @ self.r.weight.to(t).T).half()
        else:
            assert MODE == 'P'
            out = torch.nn.functional.linear(flat, self.r.weight, None)
            q = torch.empty_like(out)
            self.q.operator.call(flat, self.q.qweight, self.q.scales, q)
            ref = flat.double() @ self.r.weight.double().T
            row = dict(name=self.r.name, step=step, M=flat.shape[0], checked=ref.numel(),
                       Q=gemm_stats(q, ref), R=gemm_stats(out, ref),
                       q_vs_r_rel_l2=float((q.double() - out.double()).norm() / out.double().norm()))
            if step == 0:  # contract reference is CPU FP64; cross-check the GPU FP64 one at prefill
                cpu = flat.cpu().double() @ self.r.weight.cpu().double().T
                row['gpu_vs_cpu_fp64_rel'] = float((ref.cpu() - cpu).norm() / cpu.norm())
            PROBE.append(row)
        if self.r.bias is not None:
            out = out + self.r.bias
        return out.reshape(*x.shape[:-1], self.r.n)


def gate(c, r):
    """E017's unchanged full-model gate, plus the same bound on per-position mean-removed logits."""
    c, r = c.astype(np.float64).reshape(len(c), -1), r.astype(np.float64).reshape(len(r), -1)
    d = np.abs(c - r)
    bad = d > .03 + .01 * np.abs(r)
    cc, rc = c - c.mean(1, keepdims=True), r - r.mean(1, keepdims=True)
    return dict(bad=int(bad.sum()), bad_by_position=bad.sum(1).tolist(), max_abs=float(d.max()),
                rel_l2=float(np.linalg.norm(c - r) / np.linalg.norm(r)),
                rel_l2_by_position=(np.linalg.norm(c - r, axis=1) / np.linalg.norm(r, axis=1)).tolist(),
                centered_bad=int((np.abs(cc - rc) > .03 + .01 * np.abs(rc)).sum()),
                bitwise_equal=bool(np.array_equal(c, r)))


def main():
    global MODE
    out = Path(sys.argv[1])
    gen = int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=False)
    results = E017 / 'results/e017_20261109'
    work = I.Workload(results, 'R')
    op = I.Operator()
    manifest = json.loads((results / 'artifacts/manifest.json').read_text())
    duals = []
    for row in work.identity['coverage']:
        name = row['name']
        r = work.model.get_submodule(name)
        data = I.read_artifact(results / 'artifacts' / f'{name}.npz', manifest['layers'][name])
        q = I.Linear(name, r.bias, data, 'Q', op).eval()
        parent, leaf = name.rsplit('.', 1)
        duals.append(Dual(r, q))
        setattr(work.model.get_submodule(parent), leaf, duals[-1])
    assert len(duals) == 196
    for layer in work.model.model.layers:
        layer.register_forward_hook(lambda m, a, o: HID.append((o[0] if isinstance(o, tuple) else o).float().cpu()))
    runner = Runner(work, 'R', 128)
    runs = {}
    for label in ('R', 'Q', 'Q2', 'F32', 'F64', 'P', 'R2'):
        MODE = label.rstrip('2')
        HID.clear()
        for d in duals:
            d.calls = 0
        row, logits = runner.request('primary', generated=gen, capture_logits=True)
        assert all(d.calls == gen for d in duals)
        runs[label] = dict(tokens=row['output_ids'], logits=logits.numpy().copy(), hidden=list(HID))
        print('RAN', label, flush=True)
    R = runs['R']
    summary = dict(generated=gen, gpu=work.identity['gpu'], workspace=op.audit(),
                   tokens={k: v['tokens'] for k, v in runs.items()},
                   tokens_equal_R={k: v['tokens'] == R['tokens'] for k, v in runs.items()},
                   versus_R={k: gate(v['logits'], R['logits']) for k, v in runs.items() if k != 'R'},
                   versus_F64={k: gate(runs[k]['logits'], runs['F64']['logits']) for k in ('R', 'Q', 'F32')})
    if gen == 64:  # anchor fresh runs to the preserved E017 tensors
        for v in ('R', 'Q'):
            saved = np.load(results / f'validation_{v}_128_primary_0-output.npz')['logits']
            summary[f'fresh_{v}_versus_saved_E017_{v}'] = gate(runs[v]['logits'], saved)
    # Block-output divergence from R: [forward call (0=prefill), block]
    hidden = {}
    for k in ('Q', 'F32', 'F64', 'P', 'R2'):
        rel = [float((a - b).norm() / b.norm()) for a, b in zip(runs[k]['hidden'], R['hidden'])]
        hidden[k] = np.array(rel).reshape(gen, 28)
    summary['block_output_rel_l2_vs_R'] = {k: dict(prefill_by_block=v[0].tolist(),
        last_block_by_call=v[:, -1].tolist(), max=float(v.max())) for k, v in hidden.items()}
    probe_bad = [p for p in PROBE if p['Q']['bad'] or p['R']['bad']]
    summary['probe'] = dict(calls=len(PROBE), checked=sum(p['checked'] for p in PROBE),
        Q_bad_calls=sum(bool(p['Q']['bad']) for p in PROBE), R_bad_calls=sum(bool(p['R']['bad']) for p in PROBE),
        Q_max_rel_l2=max(p['Q']['rel_l2'] for p in PROBE), R_max_rel_l2=max(p['R']['rel_l2'] for p in PROBE),
        Q_median_rel_l2=float(np.median([p['Q']['rel_l2'] for p in PROBE])),
        R_median_rel_l2=float(np.median([p['R']['rel_l2'] for p in PROBE])),
        max_gpu_vs_cpu_fp64_rel=max(p.get('gpu_vs_cpu_fp64_rel', 0) for p in PROBE), failing=probe_bad[:50])
    assert len(PROBE) == 196 * gen
    I.save(out / 'probe.json', PROBE)
    np.savez_compressed(out / 'hidden.npz', **hidden)
    np.savez_compressed(out / 'logits.npz', **{k: v['logits'] for k, v in runs.items()})
    I.save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
