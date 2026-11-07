#!/usr/bin/env python3
"""Release exactly the winning rectangle from the predeclared timing gate."""
import json
from pathlib import Path
import sys

summary=json.loads(Path(sys.argv[1]).read_text())
assert summary['timing_rows']==16 and summary['timed_samples']==1600
shape=summary['profile_shape']
assert shape in ([4096,11008],[11008,4096]),'Timing gate failed, no profiling'
gate=next(x for x in summary['rectangles'] if [x['K'],x['N']]==shape)
assert gate['passes'] and len(gate['no_flush_gain_pct'])==2
assert all(x>=5 for x in gate['no_flush_gain_pct'])
print(*shape,sep='\n')
