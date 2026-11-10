"""Check CPU quantization bits against the unchanged project's C++ implementation."""
import argparse
import ctypes as ct
import numpy as np
from quantize import quantize
p=argparse.ArgumentParser();p.add_argument('library');a=p.parse_args()
lib=ct.CDLL(a.library)
lib.e017_quant_contract.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]+[ct.c_void_p]*3
lib.e017_quant_contract.restype=ct.c_int
rng=np.random.default_rng(17)
for kind in ('ties','random','zeros','tiny'):
    w=(rng.standard_normal((256,64))*.07).astype(np.float16)
    if kind=='ties':
        w.fill(0);w[:8,0]=[-7,-.5,.5,1.5,2.5,-1.5,-2.5,7]
    elif kind=='zeros':w.fill(0)
    elif kind=='tiny':w=(w*np.float16(1e-5)).astype(np.float16)
    actual=quantize(w)
    packed=np.empty((32,64),np.uint32);scales=np.empty((2,64),np.float16);reconstructed=np.empty_like(w)
    assert lib.e017_quant_contract(w.ctypes.data,256,64,packed.ctypes.data,scales.ctypes.data,reconstructed.ctypes.data)==0
    for name,expected in [('packed',packed),('scales',scales),('reconstructed',reconstructed)]:
        assert actual[name].tobytes()==expected.tobytes(),(kind,name)
print('PASS four complete CPU quantization cases bitwise match unchanged C++ packing/scales/reconstruction')
