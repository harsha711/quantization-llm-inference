#!/usr/bin/env python3
"""Profile one verified dense INT4 Marlin launch, outside application event timing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from bench import bridge,helpers,load_marlin

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--bridge',type=Path,required=True)
p.add_argument('--M',type=int,required=True)
p.add_argument('--K',type=int,default=4096)
p.add_argument('--N',type=int,default=4096)
p.add_argument('--warmup',type=int,default=10)
a=p.parse_args()
b=bridge(a.bridge);ns=helpers(a.source);qtype=load_marlin(a.source,a.library)
m,n,k=a.M,a.N,a.K
assert min(m,n,k)>0 and k%128==0 and n%64==0 and k>=256
cpu=[np.empty((m,k),np.float16),np.empty((k,n),np.float16),np.empty((k//8,n),np.uint32),np.empty((k//128,n),np.float16),np.empty((k,n),np.float16)]
assert b.e004_generate(m,n,k,1,*[x.ctypes.data for x in cpu])==0
av=torch.from_numpy(cpu[0]).cuda()
q=torch.from_numpy(cpu[2]).cuda().view(torch.int32)
s=torch.from_numpy(cpu[3]).cuda()
packed=torch.ops._C.gptq_marlin_repack(q,k,n,4,False)
scales=ns['marlin_permute_scales'](s,k,n,128)
c=torch.empty((m,n),device='cuda',dtype=torch.float16)
locks=torch.zeros(torch.cuda.get_device_properties(0).multi_processor_count*4,device='cuda',dtype=torch.int32)
def launch():
 return torch.ops._C.marlin_gemm(av,c,packed,None,scales,None,None,None,locks,qtype,m,n,k,False,True,False)
for _ in range(a.warmup):launch()
torch.cuda.synchronize()
print(f'PROFILE M={m} K={k} N={n} warmup={a.warmup} atomic=False fp32_reduce=True',flush=True)
torch.cuda.profiler.start()
launch()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
assert torch.isfinite(c).all()
