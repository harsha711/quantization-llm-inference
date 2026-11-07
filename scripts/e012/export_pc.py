#!/usr/bin/env python3
"""Export exact report PC instances and source lines, never inferred phase times."""
import argparse
import gzip
import json
from pathlib import Path
import sys

from regions import RANGES, source_map, resolve_region, verified_sources
from check_validation import e011


def instances(action, names, mapping):
    pcs = {}
    metadata = {}
    for name in names:
        metric = action.metric_by_name(name)
        assert metric is not None, name
        metadata[name] = dict(unit=metric.unit(), description=metric.description(),
                              value=metric.value(), instances=metric.num_instances())
        is_pc = name == 'inst_executed' or name == 'smsp__pcsamp_sample_count' or name.startswith('smsp__pcsamp_warps_issue_stalled_')
        if not is_pc:
            metadata[name]['instance_values'] = [metric.value(i) for i in range(metric.num_instances())]
            continue
        assert metric.num_instances(), name
        assert metric.has_correlation_ids(), name
        ids = metric.correlation_ids()
        assert ids.num_instances() == metric.num_instances(), name
        seen = set()
        for index in range(metric.num_instances()):
            pc, value = ids.value(index), metric.value(index)
            assert type(pc) is int and pc >= 0 and pc not in seen, (name, pc)
            assert type(value) is int and value >= 0, (name, pc, value)
            seen.add(pc)
            if pc not in pcs:
                source = action.source_info(pc)
                filename, line = (source.file_name(), source.line()) if source else ('', 0)
                locations = mapping.get(pc, {}).get('locations', [])
                pcs[pc] = dict(pc=pc, primary_file=filename, primary_line=line,
                               sass=action.sass_by_pc(pc), locations=locations,
                               region=resolve_region(locations), counts={})
            pcs[pc]['counts'][name] = value
    return metadata, [pcs[key] for key in sorted(pcs)]


def main(args):
    assert not args.out.exists()
    package = Path('/opt/nvidia/nsight-compute/2026.3.1/extras/python')
    assert package.is_dir(), package
    sys.path.insert(0, str(package))
    import ncu_report
    context = ncu_report.load_report(str(args.report))
    assert context.num_ranges() == 1
    r = context.range_by_idx(0)
    assert r.num_actions() == 1
    action = r.action_by_idx(0)
    names = args.metrics.read_text().splitlines()
    assert names and len(set(names)) == len(names)
    mapping, source_name = source_map(args.source_csv)
    assert e011.kernel_identity(source_name) == e011.kernel_identity(action.name(action.NameBase_DEMANGLED))
    metadata, pcs = instances(action, names, mapping)
    assert pcs, 'No PC instances exported. Do not substitute aggregate counters.'
    result = dict(report=str(args.report), name=action.name(action.NameBase_DEMANGLED), metric_metadata=metadata,
                  source_sha256=verified_sources(args.e006), ranges=RANGES, pcs=pcs,
                  interpretation='All exported inline source locations retained, each PC counted once. Samples are not phase wall time. Conflicting caller phases remain ambiguous.')
    with gzip.open(args.out, 'wt') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(f'PASS: {len(pcs)} exact PCs exported with metric units, source lines and conservative regions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ncu', 'report', 'source-csv', 'metrics', 'e006', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
