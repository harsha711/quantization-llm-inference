"""Pure evidence checks shared by validation, timing and offline analysis."""
import math


def passes(gains):
    return len(gains) == 2 and all(math.isfinite(x) and x >= 5 for x in gains)


def traversal(series):
    cells = [(length, variant) for length in (128, 2048) for variant in ('E', 'G')]
    assert series in (1, 2)
    return cells if series == 1 else cells[::-1]


def checked(row, reference, length, generated=64):
    assert row['context'] == length and row['generated'] == generated
    assert row['decode_forwards'] == generated-1
    assert row['input_ids'] == reference['input_ids'] and row['output_ids'] == reference['output_ids']
    assert row['eos_positions'] == reference['eos_positions']
    assert row['cache_length'] == length+generated-1 and row['static_capacity'] == length+64
    assert row['cache_class'] == 'StaticCache' and row['cache_pointers_stable']
    assert row['variant'] in ('E', 'G')
    assert row['graph_replays'] == (generated-1 if row['variant'] == 'G' else 0)
    for key in ('decode_s', 'ttft_s', 'e2e_s', 'preparation_s', 'prefill_to_first_s',
                'post_first_boundary_s', 'output_s'):
        assert math.isfinite(row[key]) and row[key] >= 0, key
    assert row['decode_s'] > 0 and row['ttft_s'] > 0
    total = sum(row[k] for k in ('preparation_s', 'prefill_to_first_s', 'post_first_boundary_s', 'decode_s', 'output_s'))
    assert math.isclose(row['e2e_s'], total, rel_tol=1e-8, abs_tol=1e-7)
    assert math.isclose(row['ttft_s'], row['preparation_s']+row['prefill_to_first_s'], abs_tol=1e-7)
    assert 0 < row['peak_allocated'] <= row['peak_reserved'] < row['device_total']*.75
