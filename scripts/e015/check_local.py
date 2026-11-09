#!/usr/bin/env python3
"""Check replay accounting, fixed work, timing decomposition and gate boundaries."""
import copy
from checks import checked, passes, traversal

assert traversal(1)[::-1] == traversal(2)
assert passes([5, 6]) and not passes([5, 4.99]) and not passes([6, float('nan')])
ref = dict(input_ids=list(range(128)), output_ids=list(range(64)), eos_positions=[])
row = dict(**ref, context=128, generated=64, decode_forwards=63, cache_length=191,
           static_capacity=192, cache_class='StaticCache', cache_pointers_stable=True,
           variant='G', graph_replays=63, decode_s=4., ttft_s=3., e2e_s=7.3,
           preparation_s=1., prefill_to_first_s=2., post_first_boundary_s=.1, output_s=.2,
           peak_allocated=100, peak_reserved=120, device_total=1024)
checked(row, ref, 128)
for key,value in [('graph_replays', 0), ('cache_length', 129), ('cache_pointers_stable', False),
                  ('decode_s', float('nan')), ('output_ids', [0]*64), ('e2e_s', 7.), ('generated', 63)]:
    bad = copy.deepcopy(row); bad[key] = value
    try:
        checked(bad, ref, 128)
    except AssertionError:
        pass
    else:
        raise AssertionError(key)
print('PASS replay count, cache progression, pointers, timing, tokens, fixed work, reversed order and 5% gate')
