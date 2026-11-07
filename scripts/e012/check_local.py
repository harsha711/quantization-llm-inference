#!/usr/bin/env python3
"""Reject invalid report data and keep genuinely ambiguous source sites explicit."""
from pathlib import Path

from export_pc import instances
from regions import RANGES, region, resolve_region, source_map, verified_sources

assert all(end < right for (_, end, _), (right, _, _) in zip(RANGES, RANGES[1:]))
assert region('external/patched/csrc/quantization/marlin/marlin_template.h', 1662) == 'stripe_drain_barrier'
assert region('external/patched/csrc/quantization/marlin/marlin_template.h', 1767) == 'global_reduction_control'
assert region('external/patched/csrc/quantization/marlin/marlin_template.h', 151) == 'lock_coordination'
assert region('external/patched/csrc/quantization/marlin/marlin.cuh', 80) == 'async_helper_ambiguous'
assert region('unrelated/dequant.h', 50) == 'external_or_ambiguous'
assert region('', 0) == 'unmapped'
assert region('marlin_template.h', 1199) == 'unclassified'
verified_sources(Path(__file__).resolve().parents[2] / 'results/e006_20261106')

class Metric:
    def __init__(self, values=(7, 11), pcs=(16, 32), correlated=True):
        self.values, self.pcs, self.correlated = values, pcs, correlated
    def unit(self): return 'inst'
    def description(self): return 'test counter'
    def value(self, index=None): return sum(self.values) if index is None else self.values[index]
    def num_instances(self): return len(self.values)
    def has_correlation_ids(self): return self.correlated
    def correlation_ids(self): return Metric(self.pcs)

class Action:
    def __init__(self, metric): self.metric = metric
    def metric_by_name(self, name): return self.metric
    def source_info(self, pc): return None
    def sass_by_pc(self, pc): return 'NOP'

meta, pcs = instances(Action(Metric()), ['inst_executed'], {})
assert len(pcs) == 2 and sum(p['counts']['inst_executed'] for p in pcs) == 18
assert all(p['region'] == 'unmapped' for p in pcs)
for candidate in (None, Metric(pcs=(16, 16)), Metric(correlated=False),
                  Metric(values=(-1, 11)), Metric(values=(float('nan'), 11))):
    try:
        instances(Action(candidate), ['inst_executed'], {})
    except (AssertionError, TypeError, ValueError):
        pass
    else:
        raise AssertionError('corrupt evidence accepted')
print('PASS: source hash/region boundaries, ambiguous sites and PC counter trust-boundary checks')

assert resolve_region([dict(region='main_loop'), dict(region='dtype_helper_ambiguous')]) == 'main_loop'
assert resolve_region([dict(region='main_loop'), dict(region='epilogue')]) == 'phase_ambiguous'
assert resolve_region([dict(region='lock_coordination'), dict(region='global_reduction_control')]) == 'lock_coordination'
root = Path(__file__).resolve().parents[2] / 'results/e012_source_c54_20261107'
for m, expected in ((4, 1960), (128, 4072)):
    pcs, name = source_map(root / f'source_M{m}.csv')
    assert len(pcs) == expected and 'Marlin<' in name
    assert any(len(p['locations']) > 1 for p in pcs.values())
    assert any(resolve_region(p['locations']) == 'phase_ambiguous' for p in pcs.values())
print('PASS: real source exports retain multiple inline contexts without double-counting PCs')
