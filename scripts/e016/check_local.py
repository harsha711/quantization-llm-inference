#!/usr/bin/env python3
"""Reject broken replay/state/timing records without a GPU."""
import copy
from checks import checked, passes, traversal

assert traversal(1)[::-1] == traversal(2)
assert passes([5, 6]) and not passes([5, 4.99]) and not passes([6, float('nan')])
ref = dict(input_ids=list(range(128)), output_ids=list(range(64)), eos_positions=[], cache_state=None)
row = dict(**ref, context=128, generated=64, decode_forwards=63, cache_length=191,
    static_capacity=192, cache_class='StaticCache', cache_pointers_stable=True, storage_stable=True,
    variant='H', graph_replays=1, decode_s=4., ttft_s=3., e2e_s=7.3,
    preparation_s=1., prefill_to_first_s=2., post_first_boundary_s=.1, output_s=.2,
    peak_allocated=100, peak_reserved=120, device_total=1024, graph_setup_s=.1,
    cache_lengths=[191]*28, positions_seen=list(range(128, 191)), feedback=list(range(63)),
    restoration=dict(post_prefill_bits_restored=True, tail_zero=True, pointers_stable=True,
                     cache_lengths=[128]*28, input_token=0))
checked(row, ref, 128)
for key, value in [('graph_replays', 63), ('positions_seen', [128]*63), ('feedback', [0]*63),
    ('storage_stable', False), ('cache_lengths', [191]*27+[190]), ('decode_s', float('nan')),
    ('output_ids', [0]*64), ('e2e_s', 7.), ('generated', 63),
    ('restoration', dict(row['restoration'], post_prefill_bits_restored=False))]:
    bad = copy.deepcopy(row)
    bad[key] = value
    try:
        checked(bad, ref, 128)
    except AssertionError:
        pass
    else:
        raise AssertionError(key)
print('PASS ten state/replay/timing mutations, reverse traversal and fixed 5% gate')
