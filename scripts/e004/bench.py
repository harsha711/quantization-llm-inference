#!/usr/bin/env python3
"""Direct vLLM Marlin vs retained CUDA kernels, identical host-generated operands.

No vLLM package import: load its released operator library and exact source helpers.
CUDA-graph event nodes remove Python/ctypes dispatch gaps for every selector.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PIN = "4ba787ec9959e566d3b587a796dff07673324da5"
NAMES = ["naive", "inblock_specialized", "inblock_m2", "inblock_m4", "cublas",
         "cublas_dq", "unfused", "dequant", "inblock"]


def helpers(source):
    """Load only inspected pure packing helpers, avoiding serving dependencies."""
    ns = {"np": np, "numpy": np, "torch": torch, "GPTQ_MARLIN_TILE": 16}
    files = {
        "marlin_utils_test.py": {"get_weight_perm", "marlin_permute_weights"},
        "marlin_utils.py": {"get_scale_perms", "marlin_permute_scales"},
    }
    base = source / "vllm/model_executor/layers/quantization/utils"
    for file, names in files.items():
        tree = ast.parse((base / file).read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in selected} == names
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(base/file), "exec"), ns)
    return ns


def unpack_project(q, k, n):
    return ((q.astype(np.uint32)[:, None, :] >> (4*np.arange(8, dtype=np.uint32))[None, :, None]) & 15).reshape(k, n)


def unpack_marlin(packed, k, n, perm):
    vals = ((packed.astype(np.uint32)[:, :, None] >> (4*np.arange(8, dtype=np.uint32))[None, None, :]) & 15).reshape(-1, 1024)
    vals = vals[:, np.argsort(perm)].reshape(k//16, n//16, 16, 16)
    return vals.transpose(0, 2, 1, 3).reshape(k, n)


def bridge(path):
    b = ct.CDLL(str(path.resolve()))
    p, i = ct.c_void_p, ct.c_int
    b.e004_generate.argtypes = [i,i,i,ct.c_uint64,p,p,p,p,p]
    b.e004_generate.restype = i
    b.e004_reference.argtypes = [p,p,i,i,i,p]
    b.e004_handle.restype = p
    b.e004_destroy.argtypes = [p]
    b.e004_l2.restype = i
    b.e004_flush.argtypes = [p,ct.c_size_t,p]
    b.e004_launch.argtypes = [i,p,p,p,p,p,p,i,i,i,p,p]
    return b


def load_marlin(source, library):
    # ScalarType has no package imports. Register it for dataclass introspection.
    spec = importlib.util.spec_from_file_location("e004_scalar_type", source/"vllm/scalar_type.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    torch.ops.load_library(str(library.resolve()))
    print("OPERATOR_SCHEMA", torch.ops._C.marlin_gemm.default._schema, flush=True)
    return mod.scalar_types.uint4b8.id


def validate(got, ref):
    g = got.float().cpu().numpy().astype(np.float64)
    rms = float(np.sqrt(np.mean(ref*ref)))
    err = np.abs(g-ref)
    bad = int(np.count_nonzero(~np.isfinite(g) | (err > 1e-3*np.abs(ref)+1e-3*rms)))
    st = dict(num_bad=bad, checked=int(g.size), rms=rms, max_abs=float(err.max()),
              rel_l2=float(np.linalg.norm(g-ref)/np.linalg.norm(ref)))
    assert bad == 0, st
    return st


def timed(launch, flush, warmup, iters, output, graph=True):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    # Allocate event objects/graph outside measurement. External event nodes ensure
    # elapsed times measure device work and exclude host dispatch and store flush.
    starts = [torch.cuda.Event(enable_timing=True, external=graph) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True, external=graph) for _ in range(iters)]
    def sequence():
        for a,z in zip(starts,stops):
            if flush is not None:
                flush()
            a.record()
            launch()
            z.record()
    setup_start = time.perf_counter()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.current_stream()):
            sequence()
        capture_s = time.perf_counter()-setup_start
        output.fill_(float("nan"))
        g.replay()
    else:
        capture_s = 0.0
        output.fill_(float("nan"))
        sequence()
    torch.cuda.synchronize()
    us = [a.elapsed_time(z)*1000 for a,z in zip(starts,stops)]
    ordered = sorted(us)
    return dict(capture_wall_s=capture_s, samples_us=us, median_us=float(np.median(us)), min_us=min(us), max_us=max(us),
                p90_us=ordered[math.floor(.9*(iters-1))])


def run(args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists(), f"Refusing to overwrite {args.out}"
    b = bridge(args.bridge)
    ns = helpers(args.source)
    qtype = load_marlin(args.source, args.library)
    handle = b.e004_handle()
    print("MAPPED_LIBRARIES", "\n".join(x for x in Path("/proc/self/maps").read_text().splitlines() if any(n in x for n in ["libcublas", "libcudart", "libe004"])), flush=True)
    stream = torch.cuda.Stream()
    l2 = b.e004_l2()
    flush_buf = torch.empty(2*l2, dtype=torch.uint8, device="cuda")
    print(json.dumps(dict(source_commit=PIN, torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                          l2_bytes=l2, stream=int(stream.cuda_stream), qtype=qtype,
                          graph_events=True, atomic=False, fp32_reduce=True)), flush=True)
    with torch.cuda.stream(stream), args.out.open("x") as out:
        for k,n in args.shapes:
            assert k % 128 == 0 and n % 64 == 0 and k >= 256
            maxm = max(args.ms)
            cpu = [np.empty((maxm,k),np.float16), np.empty((k,n),np.float16),
                   np.empty((k//8,n),np.uint32), np.empty((k//128,n),np.float16),
                   np.empty((k,n),np.float16)]
            setup_start = time.perf_counter()
            assert b.e004_generate(maxm,n,k,1,*[x.ctypes.data for x in cpu]) == 0
            generated_s = time.perf_counter()-setup_start
            operand_sha256 = {name:hashlib.sha256(x.tobytes()).hexdigest() for name,x in zip(["A","W","Q","S","W_hat"],cpu)}
            a,w,q,s,wh = [torch.from_numpy(x).to("cuda") for x in cpu]
            q = q.view(torch.int32)
            # Both formats use the same offset codes and FP16 scale bits.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ev0,ev1 = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            ev0.record()
            packed = torch.ops._C.gptq_marlin_repack(q,k,n,4,False)
            scales = ns["marlin_permute_scales"](s,k,n,128)
            ev1.record()
            torch.cuda.synchronize()
            repack_s, repack_gpu_us = time.perf_counter()-t0, ev0.elapsed_time(ev1)*1000
            perm = ns["get_weight_perm"](4).numpy()
            recovered = unpack_marlin(packed.cpu().numpy(),k,n,perm)
            codes = unpack_project(cpu[2],k,n)
            assert np.array_equal(recovered,codes), "repack lost integer codes"
            inv = np.argsort(np.array(ns["get_scale_perms"]()[0]))
            rs = scales.cpu().numpy().reshape(-1,64)[:,inv].reshape(k//128,n)
            assert np.array_equal(rs.view(np.uint16),cpu[3].view(np.uint16)), "scale bit loss"
            recovered_wh = ((recovered.astype(np.float32)-8)*rs.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
            assert np.array_equal(recovered_wh.view(np.uint16),cpu[4].view(np.uint16)), "W_hat bit loss"
            print(f"REPACK_PASS K={k} N={n} elements={k*n} generated_s={generated_s} repack_s={repack_s} repack_gpu_us={repack_gpu_us}",flush=True)
            refs = []
            t0=time.perf_counter()
            for hostw in [cpu[1],cpu[4]]:
                r=np.empty((maxm,n),np.float64)
                b.e004_reference(cpu[0].ctypes.data,hostw.ctypes.data,maxm,n,k,r.ctypes.data)
                refs.append(r)
            reference_s=time.perf_counter()-t0
            workspace = torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count*4,dtype=torch.int32,device="cuda")
            c = torch.empty((maxm,n),dtype=torch.float16,device="cuda")
            # Pre-materialize W_hat outside timed regions (unfused updates it per call).
            b.e004_launch(7,a.data_ptr(),q.data_ptr(),s.data_ptr(),w.data_ptr(),wh.data_ptr(),c.data_ptr(),maxm,n,k,stream.cuda_stream,handle)
            torch.cuda.synchronize()
            assert np.array_equal(wh.cpu().numpy().view(np.uint16),cpu[4].view(np.uint16))
            for m in args.ms:
                av,cv = a[:m],c[:m]
                def marlin():
                    return torch.ops._C.marlin_gemm(av,cv,packed,None,scales,None,None,None,workspace,qtype,m,n,k,False,True,False)
                def retained(name):
                    return lambda: b.e004_launch(NAMES.index(name),av.data_ptr(),q.data_ptr(),s.data_ptr(),w.data_ptr(),wh.data_ptr(),cv.data_ptr(),m,n,k,stream.cuda_stream,handle)
                # Preserve E002/E001 and E003/E002 full-output obligations.
                retained("inblock")(); torch.cuda.synchronize(); control=cv.clone()
                retained("inblock_specialized")(); torch.cuda.synchronize()
                assert torch.equal(cv.view(torch.int16),control.view(torch.int16))
                control=cv.clone()
                for name in ["inblock_m2","inblock_m4"]:
                    retained(name)(); torch.cuda.synchronize()
                    assert torch.equal(cv.view(torch.int16),control.view(torch.int16))
                for name in args.kernels:
                    launch = marlin if name == "marlin" else retained(name)
                    cv.fill_(float("nan"))
                    first_call_start=time.perf_counter()
                    launch();torch.cuda.synchronize()
                    first_call_wall_s=time.perf_counter()-first_call_start
                    if args.trace and name == "marlin":
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                            launch()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(str(args.out.parent/f"{args.out.stem}_M{m}_K{k}_N{n}_trace.json"))
                        gpu_names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        print("EXECUTED_KERNELS", json.dumps(gpu_names), flush=True)
                        assert any("marlin::Marlin<" in x for x in gpu_names), gpu_names
                        assert all("marlin::Marlin<" in x for x in gpu_names), gpu_names
                    stats=validate(cv,refs[0 if name=="cublas" else 1][:m])
                    for policy in args.policies:
                        flush = (lambda:b.e004_flush(flush_buf.data_ptr(),flush_buf.numel(),stream.cuda_stream)) if policy=="store_flush" else None
                        timing=timed(launch,flush,args.warmup,args.iters,cv,not args.eager)
                        # Validate final output too, including graph replay.
                        validate(cv,refs[0 if name=="cublas" else 1][:m])
                        row=dict(kernel=name,M=m,K=k,N=n,group=128,seed=1,policy=policy,warmup=args.warmup,iters=args.iters,
                                 method="eager_events" if args.eager else "graph_events",source_commit=PIN,pass_check=True,
                                 repack_codes_match=True,repack_scale_bits_match=True,reconstructed_elements=k*n,
                                 first_call_wall_s=first_call_wall_s,operand_sha256=operand_sha256,generated_s=generated_s,repack_wall_s=repack_s,repack_gpu_us=repack_gpu_us,reference_s=reference_s,
                                 full_control_match=True,**stats,**timing)
                        out.write(json.dumps(row)+"\n");out.flush()
                        print(f"RESULT {name} M={m} K={k} N={n} {policy} median={timing['median_us']:.3f} p90={timing['p90_us']:.3f} error={stats['rel_l2']:.6g}",flush=True)
    b.e004_destroy(handle)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--bridge",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--shapes",nargs="+",default=["4096,4096","4096,11008","11008,4096"])
    p.add_argument("--ms",nargs="+",type=int,default=[1,2,4,8,16,32,64,128])
    p.add_argument("--kernels",nargs="+",choices=["marlin"]+[n for n in NAMES if n!="dequant"],default=["marlin"]+NAMES[:7])
    p.add_argument("--policies",nargs="+",choices=["store_flush","no_flush"],default=["store_flush","no_flush"])
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--iters",type=int,default=100)
    p.add_argument("--eager",action="store_true")
    p.add_argument("--trace",action="store_true")
    a=p.parse_args()
    a.shapes=[tuple(map(int,s.split(","))) for s in a.shapes]
    assert a.iters>0 and a.warmup>=0 and min(a.ms)>0
    run(a)
