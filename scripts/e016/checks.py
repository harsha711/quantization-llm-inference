"""Evidence gates for the single authorized graph-boundary comparison."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e015.checks import passes, checked as token_checked


def traversal(series):
    assert series in (1, 2)
    cells = [(length, variant) for length in (128, 2048) for variant in ('G', 'H')]
    return cells if series == 1 else cells[::-1]


def checked(row, reference, length, generated=64):
    # Reuse the original fixed-work/timing/numerical-output checks unchanged.
    adapted = dict(row, variant='G', graph_replays=generated-1)
    token_checked(adapted, reference, length, generated)
    assert row['variant'] in ('G', 'H')
    assert row['graph_replays'] == (generated-1 if row['variant'] == 'G' else 1)
    assert row['storage_stable']
    assert row['positions_seen'] == list(range(length, length+generated-1))
    assert row['feedback'] == row['output_ids'][:-1]
    assert row['cache_lengths'] == [length+generated-1]*28
    assert math.isfinite(row['graph_setup_s']) and row['graph_setup_s'] > 0
    if row['variant'] == 'H':
        restore = row['restoration']
        assert restore['post_prefill_bits_restored'] and restore['tail_zero'] and restore['pointers_stable']
        assert restore['cache_lengths'] == [length]*28
    if row['cache_state'] is not None:
        assert row['cache_state'] == reference['cache_state']
