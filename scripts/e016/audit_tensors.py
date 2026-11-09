#!/usr/bin/env python3
"""Independently inspect retained NPY payloads with the Python standard library."""
import argparse
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zipfile


def npy(data):
    assert data[:6] == b'\x93NUMPY'
    version = data[6:8]
    size = 2 if version == b'\x01\x00' else 4
    offset = 8+size
    header_size = int.from_bytes(data[8:offset], 'little')
    header = ast.literal_eval(data[offset:offset+header_size].decode().strip())
    assert header['descr'] == '<f2' and not header['fortran_order'], header
    raw = data[offset+header_size:]
    assert len(raw) == math.prod(header['shape'])*2
    return header['shape'], raw


def audit(path, expected):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, path
        shape, got = npy(z.read('candidate.npy'))
        ref_shape, ref = npy(z.read('reference.npy'))
    assert shape == ref_shape and math.prod(shape) == expected['logits_checked']
    assert hashlib.sha256(got).hexdigest() == expected['candidate_sha256']
    assert hashlib.sha256(ref).hexdigest() == expected['reference_sha256']
    for raw in (got, ref):
        values = array.array('H', raw)
        if sys.byteorder != 'little':
            values.byteswap()
        assert not any(v & 0x7c00 == 0x7c00 for v in values), path
    assert expected['finite'] and expected['bitwise_equal'] == (got == ref)
    if got == ref:
        assert expected['max_abs'] == expected['relative_l2'] == expected['bad_logits'] == 0
    else:
        bad, delta2, ref2, maximum = 0, 0., 0., 0.
        for (a,), (b,) in zip(struct.iter_unpack('<e', got), struct.iter_unpack('<e', ref)):
            delta = abs(a-b)
            bad += delta > .03+.01*abs(b)
            delta2 += delta*delta
            ref2 += b*b
            maximum = max(maximum, delta)
        assert bad == expected['bad_logits'] == 0 and maximum == expected['max_abs']
        assert math.isclose(math.sqrt(delta2/ref2), expected['relative_l2'], rel_tol=1e-6, abs_tol=1e-12)
        assert expected['relative_l2'] <= .003


def main(root, stage):
    paths = sorted(root.glob(stage+'_*_*-errors.json'))
    assert paths, stage
    for path in paths:
        expected = json.loads(path.read_text())
        archive = path.with_name(path.name.replace('-errors.json', '.npz'))
        audit(archive, expected)
    print('PASS independent full tensor CRC/shape/hash/finite/numerical audit:', len(paths), 'comparisons')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('smoke', 'validation'), required=True)
    a = p.parse_args()
    main(a.root, a.stage)
