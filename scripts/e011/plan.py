"""Fixed E011 scope and graph-node selection, independent of CUDA imports."""
SHAPES = [(4096, 11008), (11008, 4096)]
MS = [4, 128]
NAMES = ['marlin', 'cublas_dq']
WARMUP = 10
CYCLES = 100
MATRICES = 4


def target(kernels_by_matrix):
    """Select all kernels of matrix 3 in cycle 99, never a latency-based subset."""
    assert len(kernels_by_matrix) == MATRICES
    assert all(names and names == kernels_by_matrix[0] for names in kernels_by_matrix)
    count = len(kernels_by_matrix[-1])
    return dict(launch_skip=(CYCLES * MATRICES - 1) * count,
                launch_count=count, target_cycle=CYCLES - 1, matrix_position=MATRICES - 1)


def self_test():
    assert target([['gemm']] * 4)['launch_skip'] == 399
    assert target([['gemm', 'reduce']] * 4)['launch_skip'] == 798
    assert target([['gemm', 'reduce']] * 4)['launch_count'] == 2
    for invalid in ([], [[]] * 4, [['a'], ['a'], ['a'], ['b']], [['a']] * 3):
        try:
            target(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError(invalid)


if __name__ == '__main__':
    self_test()
    print('PASS: fixed final-operation selection, complete multi-kernel coverage and mismatch rejection')
