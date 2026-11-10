#!/usr/bin/env python3
"""Tiny deterministic arithmetic/packing checks; no model or CUDA dependency."""
import hashlib

import numpy as np

from quantize import quantize, tensor_sha


def rejected(value, exception):
    try:
        quantize(value)
    except exception:
        return
    raise AssertionError(f'accepted malformed input: {type(value)}')


def main():
    weight = np.zeros((256, 64), dtype=np.float16)
    weight[:8, 0] = [-7, -.5, .5, 1.5, 2.5, -1.5, -2.5, 7]
    weight[8:11, 0] = [-.0001, -.0, .0001]
    weight[128, 1] = np.float16(1)
    weight[129, 1] = np.float16(.3)
    weight[128, 2] = np.nextafter(np.float16(0), np.float16(1))
    original = weight.copy()
    result = quantize(weight)
    assert np.array_equal(weight.view(np.uint16), original.view(np.uint16))
    assert result['scales'][0, 0] == np.float16(1)
    assert result['codes'][:8, 0].tolist() == [1, 8, 8, 10, 10, 6, 6, 15]
    assert result['reconstructed'][1, 0].view(np.uint16) == 0  # negative half tie
    assert np.all(result['reconstructed'][8:11, 0].view(np.uint16) == 0)
    assert result['packed'][0, 0] == np.uint32(0xF66AA881)
    assert result['scales'][1, 1].view(np.uint16) == np.float16(np.float32(1) / np.float32(7)).view(np.uint16)
    assert result['scales'][1, 2] == 0
    stored_scale = np.float32(result['scales'][1, 1])
    expected_code = int(np.rint(np.float32(weight[129, 1]) / stored_scale)) + 8
    assert result['codes'][129, 1] == expected_code
    assert result['reconstructed'][129, 1] == np.float16(np.float32(expected_code - 8) * stored_scale)
    assert np.all(result['codes'][:, 2:] == 8)
    assert np.all(result['reconstructed'][:, 2:] == 0)
    assert result['codes'].min() >= 1 and result['codes'].max() <= 15
    # Independent scalar reconstruction and unpack, including every zero column.
    for row in range(256):
        for column in range(64):
            code = int(result['packed'][row // 8, column]) >> (4 * (row % 8)) & 15
            assert code == int(result['codes'][row, column])
            expected = np.float16(np.float32(code - 8) * np.float32(result['scales'][row // 128, column]))
            assert expected.view(np.uint16) == result['reconstructed'][row, column].view(np.uint16)
    assert result['hashes']['weight'] == hashlib.sha256(weight.tobytes(order='C')).hexdigest()
    transposed = weight.T.copy().T
    again = quantize(transposed)
    assert again['hashes'] == result['hashes']
    assert tensor_sha(result['packed']) == result['hashes']['packed']
    rejected(weight.astype(np.float32), TypeError)
    rejected(weight.tolist(), TypeError)
    rejected(weight[:, 0], ValueError)
    rejected(np.zeros((128, 63), np.float16), ValueError)
    rejected(np.zeros((127, 64), np.float16), ValueError)
    rejected(np.zeros((0, 64), np.float16), ValueError)
    for bad in [np.nan, np.inf, -np.inf]:
        malformed = weight.copy()
        malformed[-1, -1] = bad
        rejected(malformed, ValueError)
    print('PASS: E017 nearest-even, zero/underflow scales, FP16 reconstruction, packing, hashes, deterministic strided input and malformed-input rejection')


if __name__ == '__main__':
    main()
