"""Small CPU check of preserved half-precision payload parsing and trust gates."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/e016'))
from audit_tensors import audit, npy

header = repr(dict(descr='<f2', fortran_order=False, shape=(2,))).encode()+b'\n'
raw = struct.pack('<ee', 1., 2.)
value = b'\x93NUMPY\x01\x00'+struct.pack('<H', len(header))+header+raw
assert npy(value) == ((2,), raw)
with tempfile.TemporaryDirectory() as temp:
    path = Path(temp)/'payload.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        for name in ('candidate.npy', 'reference.npy'):
            archive.writestr(name, value)
    sha = hashlib.sha256(raw).hexdigest()
    expected = dict(logits_checked=2, candidate_sha256=sha, reference_sha256=sha,
        finite=True, bitwise_equal=True, max_abs=0, relative_l2=0, bad_logits=0)
    audit(path, expected)
    expected['candidate_sha256'] = '0'*64
    try:
        audit(path, expected)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged hash accepted')
print('PASS half payload parsing, complete numerical audit and damaged-hash rejection')
