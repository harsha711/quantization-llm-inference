#!/usr/bin/env python3
"""E017 fixed-G128 project quantization of a CPU FP16 W[K,N].

Returns codes, project-packed words, stored scales and reconstructed FP16 weights.
Persistent outputs cost (3.5 + 2/128)*K*N bytes beyond the input. Arithmetic and
hashing scratch are bounded to 128 rows; callers should process one layer at a
time and release unused representations before loading the next layer.
"""
import hashlib

import numpy as np


GROUP = 128


def tensor_sha(array):
    """SHA256 of row-major little-endian payload, excluding shape/dtype metadata."""
    digest = hashlib.sha256()
    dtype = array.dtype.newbyteorder('<')
    for start in range(0, array.shape[0], GROUP):
        chunk = np.ascontiguousarray(array[start:start + GROUP], dtype=dtype)
        digest.update(memoryview(chunk).cast('B'))
    return digest.hexdigest()


def quantize(weight):
    """Apply docs/CONTRACT.md exactly; this function never mutates ``weight``."""
    if not isinstance(weight, np.ndarray) or weight.dtype != np.dtype(np.float16):
        raise TypeError('weight must be a native-endian numpy float16 array')
    if weight.ndim != 2:
        raise ValueError('weight must have shape [K,N]')
    k, n = weight.shape
    if k <= 0 or n <= 0 or k % GROUP or n % 64:
        raise ValueError('E017 requires positive K%128=0 and N%64=0')
    codes = np.empty((k, n), dtype=np.uint8)
    packed = np.zeros((k // 8, n), dtype=np.uint32)
    scales = np.empty((k // GROUP, n), dtype=np.float16)
    reconstructed = np.empty_like(weight, order='C')
    for start in range(0, k, GROUP):
        block = weight[start:start + GROUP].astype(np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f'nonfinite input in K group starting at {start}')
        stored = (np.max(np.abs(block), axis=0) / np.float32(7)).astype(np.float16)
        scales[start // GROUP] = stored
        scale32 = stored.astype(np.float32)
        np.divide(block, scale32, out=block, where=scale32 > 0)
        block[:, scale32 == 0] = 0
        np.rint(block, out=block)
        np.clip(block, -7, 7, out=block)
        group_codes = codes[start:start + GROUP]
        np.add(block, np.float32(8), out=group_codes, casting='unsafe')
        # Reconstruct integer codes so a rounded negative zero becomes code 8's +0.
        np.subtract(group_codes, np.float32(8), out=block)
        np.multiply(block, scale32, out=block)
        with np.errstate(over='ignore'):
            reconstructed[start:start + GROUP] = block
        group_packed = packed[start // 8:(start + GROUP) // 8]
        for nibble in range(8):
            group_packed |= group_codes[nibble::8].astype(np.uint32) << (4 * nibble)
        for nibble in range(8):
            decoded = ((group_packed >> (4 * nibble)) & 15).astype(np.uint8)
            if not np.array_equal(decoded, group_codes[nibble::8]):
                raise AssertionError('project packed-code roundtrip failed')
    outputs = dict(codes=codes, packed=packed, scales=scales, reconstructed=reconstructed)
    outputs['hashes'] = dict(weight=tensor_sha(weight), **{
        name: tensor_sha(value) for name, value in outputs.items()
    })
    return outputs
