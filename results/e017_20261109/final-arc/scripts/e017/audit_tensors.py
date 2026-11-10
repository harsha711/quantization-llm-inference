"""Independent CPU payload/CRC/hash and numerical audit of preserved E017 tensors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile
import numpy as np


def sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def payload(path):
    with zipfile.ZipFile(path) as archive: assert archive.testzip() is None,path
    with np.load(path,allow_pickle=False) as archive:return {name:archive[name] for name in archive.files}


def close(actual,recorded):
    if not math.isfinite(float(actual)):
        assert recorded is None
    else: assert recorded is not None and math.isclose(float(actual),recorded,rel_tol=1e-10,abs_tol=1e-12),(actual,recorded)


def logits(path,stats,allow_failed):
    arrays=payload(path);got,ref=arrays['candidate'],arrays['reference']
    assert got.dtype==ref.dtype==np.float16 and got.shape==ref.shape
    assert sha(got)==stats['candidate_sha256'] and sha(ref)==stats['reference_sha256']
    finite=bool(np.isfinite(got).all() and np.isfinite(ref).all())
    assert finite==stats['finite'] and bool(np.array_equal(got.view(np.uint16),ref.view(np.uint16)))==stats['bitwise_equal']
    delta=np.abs(got.astype(np.float32)-ref.astype(np.float32))
    bad=~np.isfinite(got)|~np.isfinite(ref)|(delta>np.float32(.03)+np.float32(.01)*np.abs(ref.astype(np.float32)))
    assert int(bad.sum())==stats['bad_logits'] and ref.size==stats['logits_checked']
    denominator=np.linalg.norm(ref.astype(np.float64).ravel())
    numerator=np.linalg.norm(delta.astype(np.float64).ravel())
    norm=numerator/denominator if denominator else (0. if not numerator else float('inf'))
    close(delta.max(),stats['max_abs']);close(norm,stats['relative_l2'])
    passed=finite and not bad.any() and math.isfinite(norm) and norm<=.003
    if not allow_failed:assert passed,(str(path),stats)
    failed=path.with_name(path.name.replace('-comparison.npz','-failed-indices.npz'))
    if failed.exists():assert np.array_equal(payload(failed)['indices'],np.argwhere(bad))
    return passed


def audit(root,allow_failed=False):
    count=failed=gemms=0
    for path in sorted(root.glob('*-comparison.npz')):
        stats=json.loads(path.with_name(path.name.replace('-comparison.npz','-errors.json')).read_text())
        # Original versus reconstructed is a separate descriptive comparison.
        observational='-versus-original-' in path.name
        passed=logits(path,stats,allow_failed or observational)
        failed+=not passed and not observational;count+=1
    for path in sorted(root.glob('*_*-outputs.npz')):
        statpath=path.with_name(path.name.replace('-outputs.npz','.json'))
        if not statpath.exists():continue
        row=json.loads(statpath.read_text());arrays=payload(path)
        got,ref,a=arrays['candidate'],arrays['reference'],arrays['activation']
        assert got.dtype==a.dtype==np.float16 and ref.dtype==np.float64 and got.shape==ref.shape
        assert sha(got)==row['candidate_sha256'] and sha(ref)==row['reference_sha256'] and sha(a)==row['activation_sha256']
        err=np.abs(got.astype(np.float64)-ref);rms=np.sqrt(np.mean(ref*ref))
        finite=bool(np.isfinite(got).all() and np.isfinite(ref).all())
        bad=~np.isfinite(got)|(err>1e-3*np.abs(ref)+1e-3*rms)
        assert finite==row['finite'] and int(bad.sum())==row['bad'] and got.size==row['checked']
        close(err.max(),row['max_abs']);close(rms,row['rms']);close(np.linalg.norm(err)/np.linalg.norm(ref),row['rel_l2'])
        passed=finite and not bad.any()
        if not allow_failed:assert passed,(str(path),row)
        failed+=not passed;gemms+=1
    # Check every independently saved validation output against its comparison hash.
    outputs=0
    for path in sorted(root.glob('validation_*-output.npz')):
        array=payload(path)['logits'];assert array.shape==(64,1,151936) and array.dtype==np.float16
        stats=json.loads(path.with_name(path.name.replace('-output.npz','-repeat-errors.json')).read_text())
        assert sha(array)==stats['candidate_sha256'];outputs+=1
    artifacts=0
    manifest_path=root/'artifacts/manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        perm_path=manifest_path.parent/'packing-permutations.npz'
        assert hashlib.sha256(perm_path.read_bytes()).hexdigest()==manifest['packing_permutations_sha256']
        perm=payload(perm_path)
        assert np.array_equal(np.sort(perm['weight']),np.arange(1024))
        assert np.array_equal(np.sort(perm['scales']),np.arange(64))
        for name,meta in manifest['layers'].items():
            path=manifest_path.parent/(name+'.npz')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['file_sha256']
            values=payload(path);k,n=meta['K'],meta['N']
            for key in ('packed','scales','marlin_packed','marlin_scales'):assert sha(values[key])==meta['hashes'][key]
            q=values['packed'].astype(np.uint32)
            codes=((q[:,None,:]>>(4*np.arange(8,dtype=np.uint32))[None,:,None])&15).astype(np.uint8).reshape(k,n)
            assert codes.min()>=1 and codes.max()<=15 and sha(codes)==meta['hashes']['codes']
            mp=values['marlin_packed'].astype(np.uint32)
            raw=((mp[:,:,None]>>(4*np.arange(8,dtype=np.uint32))[None,None,:])&15).astype(np.uint8).reshape(-1,1024)
            recovered=raw[:,np.argsort(perm['weight'])].reshape(k//16,n//16,16,16).transpose(0,2,1,3).reshape(k,n)
            assert np.array_equal(recovered,codes)
            scales=values['marlin_scales'].reshape(-1,64)[:,np.argsort(perm['scales'])].reshape(k//128,n)
            assert np.array_equal(scales.view(np.uint16),values['scales'].view(np.uint16))
            wh=((codes.astype(np.float32)-8)*scales.repeat(128,axis=0).astype(np.float32)).astype(np.float16)
            assert np.isfinite(wh).all() and sha(wh)==meta['hashes']['reconstructed']
            artifacts+=1
        assert artifacts==196
    assert count or gemms,'no tensor evidence'
    return dict(logit_comparisons=count,gemm_comparisons=gemms,full_request_outputs=outputs,quantized_artifacts=artifacts,
        failed_numerical_gates=failed,status='payload/statistics audit complete; failed gates remain failures' if failed else 'payload and numerical gates pass')


def selfcheck():
    with tempfile.TemporaryDirectory(prefix='e017-tensor-check-') as temp:
        path=Path(temp)/'tiny-comparison.npz';ref=np.ones((2,1,3),np.float16)
        np.savez_compressed(path,candidate=ref,reference=ref)
        stats=dict(candidate_sha256=sha(ref),reference_sha256=sha(ref),finite=True,bitwise_equal=True,bad_logits=0,logits_checked=6,max_abs=0.,relative_l2=0.)
        assert logits(path,stats,False)
        try:logits(path,dict(stats,candidate_sha256='0'*64),False)
        except AssertionError:pass
        else:raise AssertionError('accepted changed tensor hash')
    print('PASS tensor payload/CRC/numerical audit and damaged-hash rejection')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?');p.add_argument('--allow-failed-gates',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:selfcheck()
    else:
        result=audit(a.root,a.allow_failed_gates)
        (a.root/'tensor-audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
