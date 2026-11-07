#!/usr/bin/env python3
"""Inspect the installed report API and existing exact-binary source correlation."""
import json
from pathlib import Path
import sys

binary, report = (Path(p).resolve() for p in sys.argv[1:])
paths = [parent / 'extras/python' for parent in binary.parents]
package = next(p for p in paths if p.is_dir())
sys.path.insert(0, str(package))
import ncu_report
context = ncu_report.load_report(str(report))
assert context.num_ranges() == 1
r = context.range_by_idx(0)
assert r.num_actions() == 1
a = r.action_by_idx(0)
print(json.dumps(dict(package=str(package), report=str(report), name=a.name(),
                     action_api=[x for x in dir(a) if not x.startswith('_')]), indent=2))
for name in ('source_info', 'sass_by_pc', 'metric_names', 'metric_by_name'):
    obj = getattr(a, name, None)
    print(name, getattr(obj, '__doc__', 'not available'))
