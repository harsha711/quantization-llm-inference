#!/usr/bin/env python3
"""Marlin vs our kernels, same process, same operands, same timer, same cache policy.

docs/OURS.md, "Marlin under the read flush". Harsha's E004 (scripts/e004/bench.py,
unchanged) timed Marlin with a store flush (inflates fast kernels) or no flush (8 MB of
weights stay in the 32 MB L2). This harness adds the read-only flush, so Marlin, v1-v3
and cuBLAS FP16 are all timed reading their weights from DRAM.

Method (kept from E004 so its numbers can be reproduced as a harness check):
  * vLLM 0.33.0's released Marlin op, loaded with torch.ops.load_library (no vllm
    import); packing helpers exec'd from the pinned source (marlin-source.tar.gz).
  * Repack is verified to preserve every 4-bit code and scale bit.
  * Each timed sample = [flush] -> start event -> one launch -> stop event, captured
    in one CUDA graph and replayed, so Python dispatch gaps are not timed.
  * Every kernel's output is checked against an FP64 reference before timing and again
    after replay.
"""
import argparse
import ast
import ctypes as ct
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

OURS_IDS = {"v1": 0, "v2": 1, "v3": 2, "cublas": 3, "inblock_specialized": 4}


def helpers(source):
    """Exec only the pure packing helpers from the pinned vLLM source (as E004)."""
    ns = {"np": np, "numpy": np, "torch": torch, "GPTQ_MARLIN_TILE": 16}
    files = {
        "marlin_utils_test.py": {"get_weight_perm", "marlin_permute_weights"},
        "marlin_utils.py": {"get_scale_perms", "marlin_permute_scales"},
    }
    base = source / "vllm/model_executor/layers/quantization/utils"
    for file, names in files.items():
        tree = ast.parse((base / file).read_text())
        sel = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in sel} == names
        exec(compile(ast.Module(body=sel, type_ignores=[]), str(base / file), "exec"), ns)
    return ns


def load_marlin(source, library):
    spec = importlib.util.spec_from_file_location("ours_scalar_type", source / "vllm/scalar_type.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    torch.ops.load_library(str(library.resolve()))
    print("OPERATOR_SCHEMA", torch.ops._C.marlin_gemm.default._schema, flush=True)
    return mod.scalar_types.uint4b8.id


def bridge(path):
    b = ct.CDLL(str(path.resolve()))
    p, i = ct.c_void_p, ct.c_int
    b.ours_generate.argtypes = [i, i, i, ct.c_uint64, p, p, p, p, p]
    b.ours_generate.restype = i
    b.ours_reference.argtypes = [p, p, i, i, i, p]
    b.ours_handle.restype = p
    b.ours_destroy.argtypes = [p]
    b.ours_l2.restype = i
    b.ours_flush_store.argtypes = [p, ct.c_size_t, p]
    b.ours_flush_read.argtypes = [p, ct.c_size_t, p, p]
    b.ours_v2_splits.argtypes = [i, i, i]
    b.ours_v2_splits.restype = i
    b.ours_launch.argtypes = [i, p, p, p, p, p, i, i, i, p, p]
    return b


def unpack_project(q, k, n):
    return ((q.astype(np.uint32)[:, None, :] >> (4 * np.arange(8, dtype=np.uint32))[None, :, None]) & 15).reshape(k, n)


def unpack_marlin(packed, k, n, perm):
    vals = ((packed.astype(np.uint32)[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))[None, None, :]) & 15).reshape(-1, 1024)
    vals = vals[:, np.argsort(perm)].reshape(k // 16, n // 16, 16, 16)
    return vals.transpose(0, 2, 1, 3).reshape(k, n)


def check(got, ref):
    g = got.float().cpu().numpy().astype(np.float64)
    rms = float(np.sqrt(np.mean(ref * ref)))
    err = np.abs(g - ref)
    bad = int(np.count_nonzero(~np.isfinite(g) | (err > 1e-3 * np.abs(ref) + 1e-3 * rms)))
    rel = float(np.linalg.norm(g - ref) / np.linalg.norm(ref))
    assert bad == 0, dict(bad=bad, rel_l2=rel)
    return rel


def timed(launch, flush, warmup, iters, output):
    for _ in range(warmup):
        if flush:
            flush()
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=torch.cuda.current_stream()):
        for a, z in zip(starts, stops):
            if flush:
                flush()
            a.record()
            launch()
            z.record()
    output.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    us = sorted(a.elapsed_time(z) * 1000 for a, z in zip(starts, stops))
    return dict(median_us=float(np.median(us)), mean_us=float(np.mean(us)), min_us=us[0],
                p90_us=us[math.floor(0.9 * (iters - 1))], max_us=us[-1])


def run(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists(), f"refusing to overwrite {args.out}"
    b = bridge(args.bridge)
    ns = helpers(args.source)
    qtype = load_marlin(args.source, args.library)
    handle = b.ours_handle()  # cuBLAS math mode 16 (FP32 reductions) set in the bridge, as E004
    stream = torch.cuda.Stream()
    l2 = b.ours_l2()
    flush_buf = torch.empty(2 * l2, dtype=torch.uint8, device="cuda")  # never written
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), l2_bytes=l2,
                          sms=torch.cuda.get_device_properties(0).multi_processor_count)), flush=True)
    flushes = {
        "read_flush": lambda: b.ours_flush_read(flush_buf.data_ptr(), flush_buf.numel(), sink.data_ptr(), stream.cuda_stream),
        "store_flush": lambda: b.ours_flush_store(flush_buf.data_ptr(), flush_buf.numel(), stream.cuda_stream),
        "no_flush": None,
    }
    with torch.cuda.stream(stream), args.out.open("x") as out:
        for k, n in args.shapes:
            assert k % 128 == 0 and n % 64 == 0
            maxm = max(args.ms)
            cpu = [np.empty((maxm, k), np.float16), np.empty((k, n), np.float16),
                   np.empty((k // 8, n), np.uint32), np.empty((k // 128, n), np.float16),
                   np.empty((k, n), np.float16)]
            assert b.ours_generate(maxm, n, k, 1, *[x.ctypes.data for x in cpu]) == 0
            a, w, q, s, _ = [torch.from_numpy(x).to("cuda") for x in cpu]
            q32 = q.view(torch.int32)

            packed = torch.ops._C.gptq_marlin_repack(q32, k, n, 4, False)
            mscales = ns["marlin_permute_scales"](s, k, n, 128)
            torch.cuda.synchronize()
            perm = ns["get_weight_perm"](4).numpy()
            assert np.array_equal(unpack_marlin(packed.cpu().numpy(), k, n, perm), unpack_project(cpu[2], k, n)), "repack lost codes"
            inv = np.argsort(np.array(ns["get_scale_perms"]()[0]))
            rs = mscales.cpu().numpy().reshape(-1, 64)[:, inv].reshape(k // 128, n)
            assert np.array_equal(rs.view(np.uint16), cpu[3].view(np.uint16)), "scale bits lost"

            refs = {}
            for name, hostw in (("orig", cpu[1]), ("what", cpu[4])):
                r = np.empty((maxm, n), np.float64)
                b.ours_reference(cpu[0].ctypes.data, hostw.ctypes.data, maxm, n, k, r.ctypes.data)
                refs[name] = r
            workspace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count * 4,
                                    dtype=torch.int32, device="cuda")
            c = torch.empty((maxm, n), dtype=torch.float16, device="cuda")
            print(f"SHAPE K={k} N={n} repack and scale bits verified", flush=True)

            for m in args.ms:
                av, cv = a[:m], c[:m]

                def marlin():
                    return torch.ops._C.marlin_gemm(av, cv, packed, None, mscales, None, None, None,
                                                    workspace, qtype, m, n, k, False, True, False)

                def ours(name):
                    i = OURS_IDS[name]
                    return lambda: b.ours_launch(i, av.data_ptr(), q.data_ptr(), s.data_ptr(), w.data_ptr(),
                                                 cv.data_ptr(), m, n, k, stream.cuda_stream, handle)

                for name in args.kernels:
                    launch = marlin if name == "marlin" else ours(name)
                    ref = refs["orig" if name == "cublas" else "what"][:m]
                    cv.fill_(float("nan"))
                    launch()  # untimed: grows v1-v3 workspaces before any capture
                    torch.cuda.synchronize()
                    rel = check(cv, ref)
                    for policy in args.policies:
                        t = timed(launch, flushes[policy], args.warmup, args.iters, cv)
                        check(cv, ref)  # graph replay produced a valid result too
                        row = dict(kernel=name, M=m, K=k, N=n, group=128, seed=1, policy=policy,
                                   method="graph_events", warmup=args.warmup, iters=args.iters,
                                   rel_l2=rel, splits=b.ours_v2_splits(m, n, k) if name in ("v2", "v3") else None, **t)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(f"RESULT {name:20s} M={m:<3d} K={k} N={n} {policy:11s} "
                              f"median={t['median_us']:8.2f} mean={t['mean_us']:8.2f}", flush=True)
    b.ours_destroy(handle)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="extracted marlin-source.tar.gz")
    p.add_argument("--library", type=Path, required=True, help="vllm/_C_stable_libtorch.abi3.so")
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shapes", nargs="+", default=["4096,4096", "1536,8960", "8960,1536", "1536,1536"])
    p.add_argument("--ms", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--kernels", nargs="+", choices=["marlin", *OURS_IDS], default=["marlin", "v3", "v2", "v1", "cublas"])
    p.add_argument("--policies", nargs="+", choices=["read_flush", "store_flush", "no_flush"],
                   default=["read_flush", "store_flush", "no_flush"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()
    a.shapes = [tuple(map(int, x.split(","))) for x in a.shapes]
    run(a)
