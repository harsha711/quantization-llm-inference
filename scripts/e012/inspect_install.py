#!/usr/bin/env python3
"""Print the installed ncu wrapper and nearby package paths, with no GPU calls."""
from pathlib import Path
import sys

p = Path(sys.argv[1]).resolve()
print('RESOLVED_NCU', p)
data = p.read_bytes()
if data.startswith(b'#!'):
    print(data[:8192].decode())
for parent in list(p.parents)[:3]:
    print('NEARBY', parent)
    for child in parent.iterdir():
        if any(x in child.name.lower() for x in ('nsight', 'cuda', 'nvidia')):
            print(child)
