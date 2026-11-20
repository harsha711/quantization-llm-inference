#!/usr/bin/env python3
"""E052: run an existing model script with the 196 block projections (and any head the script builds through
`integration.Linear(..., 'Q', op)`) executed by our `tc_ladder` (E050) instead of corrected Marlin A.

t.py [--marlin] SCRIPT ARGS...           runs SCRIPT (E021 validate.py, E026 att.py, E029 long.py, E031 head.py, E038 ckpt.py) unchanged
t.py [--marlin] --eager att.py time ...  E026 att.py timing cell with the decode step executed eagerly instead of graph-replayed
                              (its anchor is the validated graph run, which the gate requires bitwise equal to eager)
Only two names in the E017 bridge change: `Operator` (tc_ladder through E050's C entry point, no workspace) and `Linear`
(variant 'Q' loads the project-layout codes/scales `packed`/`scales` instead of Marlin's repacked copies: the same codes and
scales, no repack). Inside these processes every 'Q'/'GQA'/'GQAH' label therefore denotes tc_ladder.
--marlin leaves the bridge unchanged (corrected Marlin A), so the eager Marlin comparator runs through the same eager wrapper.
"""
import ctypes as ct
import hashlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(os.environ['E017_ROOT']) / 'scripts/e017'))
import integration as I  # noqa: E402

LIB_SHA = 'a24a78fad127230f04a867e4a78e7a2a0453a6d4f0db95ea35900c7adaa971fe'  # E050 prerequisite build (c8 job 343302)


class Operator:
    """tc_ladder: M >= 64 -> tc_m32 launcher, else tc launcher (E050 bridge, unchanged). Stateless."""

    def __init__(self, maxm=None):  # maxm kept for E029 long.py, which patches this default; tc_ladder has no row bound below
        path = Path(os.environ['E052_LIB'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == LIB_SHA
        self.lib = ct.CDLL(str(path))
        self.fn = self.lib.e050_tc_ladder
        self.fn.argtypes = [ct.c_void_p] * 4 + [ct.c_int] * 4 + [ct.c_void_p]
        self.fn.restype = ct.c_int
        self.resources, self.markers = {}, False

    def call(self, a, q, s, c, dry=False):
        assert not dry
        m, k = a.shape
        n = c.shape[1]
        assert a.dtype == c.dtype == s.dtype == torch.float16 and q.dtype == torch.int32
        assert a.is_contiguous() and c.is_contiguous() and q.is_contiguous() and s.is_contiguous()
        assert a.device == c.device == q.device == s.device
        assert q.shape == (k // 8, n) and s.shape == (k // 128, n) and k % 128 == 0
        # The launchers silently run inblock_m4_vec when these fail; refuse instead (no silent fallback).
        # Grid y is ceil(M/16) (tc) or ceil(M/32) (tc_m32), at most 65535.
        assert k % 64 == 0 and n % 32 == 0 and a.data_ptr() % 16 == 0 and c.data_ptr() % 4 == 0 and 0 < m <= 16 * 65535, (m, k, n, a.data_ptr() % 16, c.data_ptr() % 4)
        rc = self.fn(a.data_ptr(), q.data_ptr(), s.data_ptr(), c.data_ptr(), m, n, k, 128, torch.cuda.current_stream().cuda_stream)
        assert rc == 0, rc

    def audit(self):
        return dict(stateless_no_workspace=True)  # nothing to guard: tc/tc_m32 use no global workspace or locks

    def tensors(self):
        return []


class Linear(I.Linear):
    def __init__(self, name, bias, data, variant, operator):
        if variant == 'Q':  # project layout qweight[k8, n] (uint32 words viewed as int32) and scales[g, n]
            data = dict(data, marlin_packed=np.ascontiguousarray(data['packed']).view(np.int32), marlin_scales=data['scales'])
        super().__init__(name, bias, data, variant, operator)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == '--marlin':
        args = args[1:]
    else:
        I.Operator, I.Linear, I.BODY = Operator, Linear, 'tc_ladder:' + LIB_SHA
    if args[0] == '--eager':
        assert args[1:3] == ['att.py', 'time']
        script = Path(__file__).resolve().parent / 'att.py'
        spec = importlib.util.spec_from_file_location('att', script)
        att = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(att)
        graph_run = att.E.GraphRunner.run
        att.E.GraphRunner.run = lambda self, prompt, graph, key='x', **kw: graph_run(self, prompt, False, key, **kw)
        a = args[3:]
        assert a[1] in att.SAVED
        with torch.inference_mode():
            att.timed(Path(a[0]), a[1], int(a[2]), a[3], int(a[4]), Path(a[5]))
    else:
        script = Path(__file__).resolve().parent / args[0]
        sys.argv = [str(script)] + args[1:]
        runpy.run_path(str(script), run_name='__main__')
