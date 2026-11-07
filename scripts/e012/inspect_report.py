#!/usr/bin/env python3
"""Inspect the installed report API and existing exact-binary source correlation."""
import json
from pathlib import Path
import sys

binary, report = (Path(p).resolve() for p in sys.argv[1:])
# The inspected CUDA wrapper selects the versioned /opt installation.
package = Path('/opt/nvidia/nsight-compute/2026.3.1/extras/python')
assert package.is_dir(), package
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
