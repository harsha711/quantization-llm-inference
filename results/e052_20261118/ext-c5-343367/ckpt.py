#!/usr/bin/env python3
"""E038: run the unchanged E017-E031 tooling on E037's calibrated checkpoint ($E038_CKPT, [N,K] int8 codes, [N,K/128] FP16 scales).

ckpt.py root NEWROOT          a copy of $E017_ROOT by symlinks whose artifacts hold the checkpoint's block tensors; every
                              existing script then runs on them with E017_ROOT=NEWROOT (GPU: the Marlin repack)
ckpt.py head ARGS...          scripts/e031/head.py ARGS with the checkpoint's head in place of the plain-rounded one
ckpt.py headconv BASE OUT     scripts/e031/headconv.py BASE OUT, likewise
Nothing is requantized: codes and scales are only re-laid-out, and the unchanged repack asserts they survive it.
"""
import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(os.environ['E017_ROOT']) / 'scripts/e017'))
import quantize as Q  # noqa: E402

CKPT = os.environ['E038_CKPT']


def project(name, weight_hash):
    """Checkpoint tensors of one layer -> the dictionary scripts/e017/quantize.py returns (project layout [K,N])."""
    with np.load(CKPT) as z:
        c, s = z[name + '.codes'], z[name + '.scales']
    assert c.dtype == np.int8 and s.dtype == np.float16 and c.shape == (s.shape[0], 128 * s.shape[1]) and np.abs(c).max() <= 7
    codes, scales = np.ascontiguousarray((c.T + np.int8(8)).astype(np.uint8)), np.ascontiguousarray(s.T)
    packed = np.zeros((codes.shape[0] // 8, codes.shape[1]), np.uint32)
    for j in range(8):
        packed |= codes[j::8].astype(np.uint32) << (4 * j)
    rec = np.empty(codes.shape, np.float16)
    for g in range(scales.shape[0]):  # by group, to bound memory for the head
        rec[128 * g:128 * g + 128] = (codes[128 * g:128 * g + 128].astype(np.float32) - 8) * scales[g].astype(np.float32)
    assert np.isfinite(rec).all()
    out = dict(codes=codes, packed=packed, scales=scales, reconstructed=rec)
    out['hashes'] = dict(weight=weight_hash, **{k: Q.tensor_sha(v) for k, v in out.items()})
    return out


def root(new):
    import torch
    import diagnose as D
    I = D.I
    old = D.E017 / 'results/e017_20261109'
    for src, dst, skip in ((D.E017, new, 'results'), (D.E017 / 'results', new / 'results', old.name), (old, new / 'results' / old.name, 'artifacts')):
        dst.mkdir()
        for p in src.iterdir():
            if p.name != skip:
                (dst / p.name).symlink_to(p)
    dest = new / 'results' / old.name / 'artifacts'
    dest.mkdir()
    manifest = json.loads((old / 'artifacts/manifest.json').read_text())
    packer = I.load_packer()
    for row in manifest['coverage']:
        name = row['name']
        data = project(name, manifest['layers'][name]['hashes']['weight'])
        I.quantize = lambda weight: data  # repack() then only re-lays-out and asserts the roundtrip
        data = I.repack(np.broadcast_to(np.float16(0), (row['K'], row['N'])), *packer)
        arrays = {k: data[k] for k in ('packed', 'scales', 'marlin_packed', 'marlin_scales')}
        with np.load(old / 'artifacts' / f'{name}.npz') as f:
            if 'bias' in f.files:
                arrays['bias'] = f['bias']
        np.savez(dest / f'{name}.npz', **arrays)
        manifest['layers'][name] = dict(K=row['K'], N=row['N'], hashes=data['hashes'], file_sha256=hashlib.sha256((dest / f'{name}.npz').read_bytes()).hexdigest())
        print('LAYER', name, flush=True)
    manifest['e038_checkpoint'] = CKPT
    I.save(dest / 'manifest.json', manifest)
    # what the FP16 baseline loads from the new root is the checkpoint's reconstruction, bit for bit
    work = I.Workload(new / 'results' / old.name, 'R')
    with np.load(CKPT) as z:
        for row in manifest['coverage']:
            name = row['name']
            want = (torch.from_numpy(z[name + '.codes']).float() * torch.from_numpy(z[name + '.scales']).float().repeat_interleave(128, 1)).half()
            assert torch.equal(work.model.get_submodule(name).weight.cpu(), want), name
    print('ROOT OK', len(manifest['coverage']), flush=True)


def head(argv):
    import torch
    spec = importlib.util.spec_from_file_location('head', HERE / 'head.py')
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)

    def quantized_head(work):
        w = np.ascontiguousarray(work.model.lm_head.weight.detach().cpu().numpy().T)  # [K,N] FP16
        data = project('lm_head', Q.tensor_sha(w))
        h.I.quantize = lambda weight: data
        return dict(h.I.repack(w, *h.I.load_packer()), K=w.shape[0], N=w.shape[1])

    h.quantized_head = quantized_head
    with torch.inference_mode():  # head.py's own dispatch
        target = Path(argv[1])
        if argv[0] == 'validate':
            target.mkdir(parents=True, exist_ok=False)
            h.validate(target, [int(c) for c in argv[2:]])
        else:
            assert argv[0] == 'time' and argv[2] in h.SAVED
            h.timed(target, argv[2], int(argv[3]), argv[4], int(argv[5]), Path(argv[6]))


if __name__ == '__main__':
    mode, argv = sys.argv[1], sys.argv[2:]
    if mode == 'root':
        root(Path(argv[0]))
    elif mode == 'head':
        head(argv)
    else:
        assert mode == 'headconv'
        Q.quantize = lambda weight: project('lm_head', Q.tensor_sha(weight))
        sys.argv = ['headconv.py'] + argv
        runpy.run_path(str(HERE / 'headconv.py'), run_name='__main__')
