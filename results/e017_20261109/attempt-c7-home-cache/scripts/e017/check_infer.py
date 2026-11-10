#!/usr/bin/env python3
"""CPU-only checks of the actual E017 scalar evidence gates; no Torch import."""
import ast
import copy
import math
from pathlib import Path


def main():
    path = Path(__file__).with_name('infer.py')
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('checked', 'accept_logits')]
    assert len(selected) == 2
    namespace = dict(math=math)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    checked, accept = namespace['checked'], namespace['accept_logits']
    row = dict(context=128, generated=64, decode_forwards=63, input_ids=[1] * 128,
        output_ids=list(range(64)), first_token=0, positions_seen=list(range(128, 191)),
        feedback=list(range(63)), static_capacity=192, cache_class='StaticCache',
        cache_lengths=[191] * 28, cache_reset_zero=[True] * 56, cache_tail_zero=True,
        cache_finite=True, storage_stable=True, logits_finite=True, variant='Q', graph_replays=0,
        decode_s=.5, ttft_s=.03, e2e_s=.55, preparation_s=.01, prefill_to_first_s=.02,
        post_first_boundary_s=.01, output_s=.01, peak_allocated=100, peak_reserved=200, device_total=1000)
    checked(row, row, 128)
    stats = dict(shape_equal=True, finite=True, bad_logits=0, relative_l2=.003)
    accept(stats)
    bad_rows = dict(cache_lengths=[190] * 28, cache_reset_zero=[False] * 56,
        positions_seen=[127] * 63, feedback=[99] * 63, first_token=1,
        cache_tail_zero=False, cache_finite=False, logits_finite=False, storage_stable=False,
        graph_replays=1, e2e_s=.6, decode_s=float('nan'), peak_reserved=800,
        output_ids=[99] * 64)
    for key, value in bad_rows.items():
        changed = copy.deepcopy(row)
        changed[key] = value
        try:
            checked(changed, row, 128)
        except AssertionError:
            continue
        raise AssertionError(f'accepted corrupt row: {key}')
    for key, value in dict(shape_equal=False, finite=False, bad_logits=1, relative_l2=.00301).items():
        changed = dict(stats, **{key: value})
        try:
            accept(changed)
        except AssertionError:
            continue
        raise AssertionError(f'accepted corrupt logits: {key}')
    for value in (float('inf'), float('nan'), None):
        try:
            accept(dict(stats, relative_l2=value))
        except AssertionError:
            continue
        raise AssertionError('accepted nonfinite norm')
    print('PASS: E017 fixed-work/cache/reset/timing gates and 21 corrupted-row/numerical cases rejected')


if __name__ == '__main__':
    main()
