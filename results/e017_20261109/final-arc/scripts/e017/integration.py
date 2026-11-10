"""E017 model bridge; unchanged E006 A through its preserved typed C ABI."""
import ctypes as ct
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/e014'))
import numpy as np
import torch
from model import Workload as OriginalWorkload, Runner as BaseRunner, tensor_sha
from scripts.e004 import bench as e004
from scripts.e005.bench import FIELDS
from quantize import quantize, tensor_sha as array_sha

BODY = 'e4eca5e30436c547358a9c8125d69526c6bb3318da034afd54ada6744181e2b7'
WHEEL = 'e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'
SHAPES = {'q_proj':(1536,1536),'k_proj':(1536,256),'v_proj':(1536,256),
          'o_proj':(1536,1536),'gate_proj':(1536,8960),'up_proj':(1536,8960),
          'down_proj':(8960,1536)}


def save(path, value):
    path = Path(path)
    assert not path.exists(), path
    text=json.dumps(value, indent=2, allow_nan=False)+'\n'
    with path.open('x') as f:f.write(text)


def inventory(model):
    rows=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear) and name.startswith('model.layers.'):
            leaf=name.rsplit('.',1)[1]; k,n=SHAPES[leaf]
            assert tuple(module.weight.shape)==(n,k),(name,module.weight.shape)
            assert (module.bias is not None)==(leaf in ('q_proj','k_proj','v_proj'))
            rows.append(dict(name=name,K=k,N=n,bias=module.bias is not None))
    assert len(rows)==196 and model.config.num_hidden_layers==28
    assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    return rows


def storage(tensors):
    unique={}
    for name,t in tensors:
        s=t.untyped_storage(); key=(str(t.device),s.data_ptr())
        row=unique.setdefault(key,dict(device=str(t.device),pointer=s.data_ptr(),bytes=s.nbytes(),names=[]))
        row['names'].append(name)
    return dict(bytes=sum(x['bytes'] for x in unique.values()),allocations=list(unique.values()))


class Operator:
    def __init__(self,maxm=2048):
        self.path=Path(os.environ['E017_LOCAL'])
        assert hashlib.sha256(self.path.read_bytes()).hexdigest()==BODY
        self.lib=ct.CDLL(str(self.path))
        self.lib.e005_call.argtypes=[ct.c_int,ct.c_int]+[ct.c_void_p]*6+[ct.c_int]*3+[ct.c_void_p,ct.POINTER(ct.c_int64)]
        self.lib.e005_call.restype=ct.c_int
        sms=torch.cuda.get_device_properties(0).multi_processor_count
        count=sms*min((maxm+15)//16*16,64)*256
        self.lock_storage=torch.full((4*sms+32,),0x12345678,dtype=torch.int32,device='cuda')
        self.locks=self.lock_storage[16:-16];self.locks.zero_()
        self.scratch_storage=torch.full((count+32,),123456.,dtype=torch.float32,device='cuda')
        self.scratch=self.scratch_storage[16:-16];self.scratch.fill_(float('nan'))
        self.maxm=maxm; self.info=(ct.c_int64*20)();self.resources={};self.markers=False

    def call(self,a,q,s,c,dry=False):
        m,k=a.shape;n=c.shape[1]
        assert a.dtype==c.dtype==s.dtype==torch.float16 and q.dtype==torch.int32
        assert a.is_contiguous() and c.is_contiguous() and q.is_contiguous() and s.is_contiguous() and 0<m<=self.maxm
        assert q.shape==(k//16,n*2) and s.shape==(k//128,n)
        assert a.device==c.device==q.device==s.device
        assert k>=256 and k%128==0 and n%64==0
        rc=self.lib.e005_call(0,int(dry),a.data_ptr(),q.data_ptr(),s.data_ptr(),c.data_ptr(),
            self.scratch.data_ptr(),self.locks.data_ptr(),m,n,k,torch.cuda.current_stream().cuda_stream,self.info)
        assert rc==0,rc
        if dry:
            row=dict(zip(FIELDS,map(int,self.info)));self.resources[f'{m},{k},{n}']=row
            return row

    def audit(self):
        result=dict(locks_zero=bool(torch.all(self.locks==0).item()),
            lock_guards=bool(torch.all(self.lock_storage[:16]==0x12345678).item() and torch.all(self.lock_storage[-16:]==0x12345678).item()),
            scratch_guards=bool(torch.all(self.scratch_storage[:16]==123456.).item() and torch.all(self.scratch_storage[-16:]==123456.).item()))
        return result

    def tensors(self):
        return [('marlin_locks',self.lock_storage),('marlin_scratch',self.scratch_storage)]


class Linear(torch.nn.Module):
    def __init__(self,name,bias,data,variant,operator):
        super().__init__();self.name=name;self.variant=variant;self.operator=operator
        self.k,self.n=data['K'],data['N']
        self.bias=bias
        if variant=='O':
            self.weight=data['weight']
        elif variant=='R':
            codes=e004.unpack_project(data['packed'],self.k,self.n)
            wh=((codes.astype(np.float32)-8)*data['scales'].repeat(128,axis=0).astype(np.float32)).astype(np.float16)
            assert array_sha(wh)==data['hashes']['reconstructed']
            self.weight=torch.nn.Parameter(torch.from_numpy(wh.T.copy()).cuda(),requires_grad=False)
        else:
            assert variant=='Q'
            self.register_buffer('qweight',torch.from_numpy(data['marlin_packed'].copy()).cuda())
            self.register_buffer('scales',torch.from_numpy(data['marlin_scales'].copy()).cuda())

    def forward(self,x):
        shape=x.shape;flat=x.reshape(-1,self.k).contiguous()
        context=torch.profiler.record_function(f'E017/linear/{self.name}/M{flat.shape[0]}') if self.operator and self.operator.markers else __import__('contextlib').nullcontext()
        with context:
            if self.variant in ('O','R'):
                out=torch.nn.functional.linear(flat,self.weight,None)
            else:
                out=torch.empty((flat.shape[0],self.n),dtype=torch.float16,device=x.device)
                self.operator.call(flat,self.qweight,self.scales,out)
            if self.bias is not None: out=out+self.bias
        return out.reshape(*shape[:-1],self.n)


def repack(weight,source,wheel):
    quant_start=time.perf_counter();data=quantize(weight)
    quantize_s=time.perf_counter()-quant_start
    assert np.isfinite(data['reconstructed']).all(),'nonfinite reconstructed actual model weight'
    ns=e004.helpers(source)
    q=torch.from_numpy(data['packed']).view(torch.int32).cuda()
    s=torch.from_numpy(data['scales']).cuda()
    start=time.perf_counter()
    packed=torch.ops._C.gptq_marlin_repack(q,*weight.shape,4,False)
    scales=ns['marlin_permute_scales'](s,*weight.shape,128)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    mp,ms=packed.cpu().numpy(),scales.cpu().numpy()
    recovered=e004.unpack_marlin(mp,*weight.shape,ns['get_weight_perm'](4).numpy())
    inv=np.argsort(np.array(ns['get_scale_perms']()[0]))
    rs=ms.reshape(-1,64)[:,inv].reshape(data['scales'].shape)
    assert np.array_equal(recovered,data['codes'])
    assert np.array_equal(rs.view(np.uint16),data['scales'].view(np.uint16))
    assert np.array_equal(((recovered.astype(np.float32)-8)*rs.repeat(128,axis=0).astype(np.float32)).astype(np.float16).view(np.uint16),data['reconstructed'].view(np.uint16))
    data.update(marlin_packed=mp,marlin_scales=ms,repack_s=elapsed,quantize_s=quantize_s)
    data['hashes'].update(marlin_packed=array_sha(mp),marlin_scales=array_sha(ms))
    return data


def load_packer():
    source=Path(os.environ['E017_SOURCE']);wheel=Path(os.environ['E017_WHEEL'])
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==WHEEL
    e004.load_marlin(source,wheel)
    return source,wheel


def read_artifact(path,metadata):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['file_sha256']
    with np.load(path,allow_pickle=False) as f: data={name:f[name] for name in f.files}
    for name in ('packed','scales','marlin_packed','marlin_scales'):
        assert array_sha(data[name])==metadata['hashes'][name]
    return dict(data,**{k:metadata[k] for k in ('K','N','hashes')})


class Workload(OriginalWorkload):
    def __init__(self,results,variant):
        start=time.perf_counter();super().__init__(results)
        self.variant=variant;self.original_identity=self.identity.copy()
        layers=inventory(self.model);self.operator=Operator() if variant=='Q' else None
        self.artifact_manifest=None;self.replace_s=0.
        if variant=='O':
            for row in layers:
                name=row['name'];old=self.model.get_submodule(name)
                new=Linear(name,old.bias,dict(K=row['K'],N=row['N'],weight=old.weight),'O',None).eval()
                parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,new)
                del old,new
        else:
            manifest=json.loads((results/'artifacts/manifest.json').read_text())
            assert manifest['original_weights_sha256']==self.weights_sha256
            assert manifest['coverage']==layers
            self.artifact_manifest=hashlib.sha256((results/'artifacts/manifest.json').read_bytes()).hexdigest()
            begin=time.perf_counter()
            for row in layers:
                name=row['name'];old=self.model.get_submodule(name)
                data=read_artifact(results/'artifacts'/f'{name}.npz',manifest['layers'][name])
                new=Linear(name,old.bias,data,variant,self.operator).eval()
                parent,leaf=name.rsplit('.',1);setattr(self.model.get_submodule(parent),leaf,new)
                del old,data,new
            torch.cuda.synchronize();self.replace_s=time.perf_counter()-begin
        load_peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        self.state_hashes={name:tensor_sha(t) for name,t in self.model.state_dict().items()}
        original=self.original_identity['weights']
        for name,p in self.model.named_parameters():
            if not any(name==r['name']+'.weight' for r in layers): assert tensor_sha(p)==original[name],name
        self.weights_sha256=hashlib.sha256(json.dumps(self.state_hashes,sort_keys=True).encode()).hexdigest()
        self.resident=storage(list(self.model.named_parameters(remove_duplicate=False))+list(self.model.named_buffers(remove_duplicate=False))+(self.operator.tensors() if self.operator else []))
        self.identity.update(parameter_count_scope='original loaded checkpoint before replacement',actual_parameter_count=sum(p.numel() for p in self.model.parameters()),load_replace_allocator_peak=load_peak,resident_scope='unique model and shared operator tensors, excludes Runner cache/buffers',variant=variant,original_weights_sha256=self.original_identity['weights_sha256'],
            weights_sha256=self.weights_sha256,artifacts_sha256=self.artifact_manifest,coverage=layers,
            replace_s=self.replace_s,model_initialization_wall_s=time.perf_counter()-start,
            resident=self.resident,corrected_body_sha256=BODY if variant=='Q' else None,
            linear_bias_policy='FP16 GEMM followed by separate FP16 bias addition in O/R/Q',graphs=False)

    def weights_unchanged(self):
        current={name:tensor_sha(t) for name,t in self.model.state_dict().items()}
        changed=[name for name in current.keys()|self.state_hashes.keys() if current.get(name)!=self.state_hashes.get(name)]
        workspace=self.operator.audit() if self.operator else None
        if changed or (workspace is not None and not all(workspace.values())):
            print('FAILED_STATE_AUDIT',dict(changed=changed,workspace=workspace),flush=True)
        assert not changed and (workspace is None or all(workspace.values()))
