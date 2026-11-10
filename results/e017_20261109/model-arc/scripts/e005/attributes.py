#!/usr/bin/env python3
"""Query launch plans and actual compiled-kernel occupancy without launching it."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import torch
from bench import FIELDS
p=argparse.ArgumentParser();p.add_argument('library',type=Path);a=p.parse_args()
lib=ct.CDLL(str(a.library.resolve()))
lib.e005_call.argtypes=[ct.c_int,ct.c_int]+[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p,ct.POINTER(ct.c_int64)]
lib.e005_call.restype=ct.c_int
out=[]
for m in (1,8,16,64):
    for mode in range(3):
        v=(ct.c_int64*20)()
        assert lib.e005_call(mode,1,*([None]*6),m,4096,4096,None,v)==0
        out.append(dict(M=m,config='ABC'[mode],**dict(zip(FIELDS,map(int,v)))))
for m in (1,8):
    points=[r for r in out if r['M']==m]
    assert len({r['kernel_address'] for r in points})==1
    assert [(r['grid'],r['dynamic_shared'],r['max_blocks_per_sm']) for r in points]==[(36,101376,1),(36,49664,2),(72,49664,2)]
for m in (16,64):
    points=[{k:v for k,v in r.items() if k!='config'} for r in out if r['M']==m]
    assert points[0]==points[1]==points[2]
print(json.dumps(out,indent=2))
