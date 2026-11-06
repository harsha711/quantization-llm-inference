#!/usr/bin/env python3
"""Use the retained E005 checker/timer with full-output hashes for paired bodies."""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path

spec=importlib.util.spec_from_file_location('e005_control',Path(__file__).resolve().parents[1]/'e005/bench.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

class CheckedCase(bench.Case):
    def check(self,name,m,repeats,graphs=False):
        result=super().check(name,m,repeats,graphs)
        result['output_sha256']=hashlib.sha256(self.c[:m].view(bench.torch.int16).cpu().numpy().tobytes()).hexdigest()
        return result

bench.Case=CheckedCase
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variant',choices=['control','candidate'],required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--ms',nargs='+',type=int,default=[1,8])
p.add_argument('--shapes',nargs='+',default=['4096,4096'])
p.add_argument('--configs',nargs='+',choices=list('ABC'),default=['A'])
p.add_argument('--patterns',nargs='+',default=['mixed'])
p.add_argument('--policies',nargs='+',default=['store_flush','no_flush'])
p.add_argument('--seed',type=int,default=1)
p.add_argument('--warmup',type=int,default=0)
p.add_argument('--iters',type=int,default=0)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--profile',action='store_true')
a=p.parse_args()
assert a.warmup>=0 and a.iters>=0 and a.repeats>0
a.shapes=[tuple(map(int,v.split(','))) for v in a.shapes]
assert all(len(v)==2 and min(v)>0 for v in a.shapes)
assert set(a.patterns)<=set(('mixed','zero_a','zero_w')) and set(a.policies)<=set(('store_flush','no_flush'))
a.source=Path(os.environ['E005_CONTROL'])/'external/pinned'
a.local=Path(os.environ['E007_RESULTS'])/a.variant/'libe005.so'
a.wheel=Path(os.environ['E005_WHEEL']);a.bridge=Path(os.environ['E005_BRIDGE'])
a.graph_check=True;a.trace=False;a.balance=1
bench.main(a)
