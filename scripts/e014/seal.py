#!/usr/bin/env python3
"""Preserve runtime sources and generated compiler artifacts after the stopped gate."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import torch
from fetch import sha

root = Path(sys.argv[1])
assert os.environ['OMPI_COMM_WORLD_SIZE'] == '1'
source = root/'runtime-source'
source.mkdir(exist_ok=False)
identity = json.loads((root/'baseline-identity.json').read_text())
records = {}
for name,info in identity['source'].items():
    original = Path(info['path'])
    assert sha(original) == info['sha256']
    shutil.copy2(original,source/(name+'.py'))
    records[name] = info
torch_root = Path(torch.__file__).parent
for rel in ('_dynamo/convert_frame.py','_dynamo/eval_frame.py','_inductor/compile_fx.py',
            '_inductor/cudagraph_trees.py','_inductor/config.py','_inductor/scheduler.py'):
    original = torch_root/rel
    records['torch/'+rel] = dict(path=str(original),sha256=sha(original))
    target = source/'torch'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(original,target)
for name in ('torch','transformers','triton','tokenizers','safetensors'):
    dist = importlib.metadata.distribution(name)
    (source/(name+'-RECORD.txt')).write_text(dist.read_text('RECORD'))
    (source/(name+'-METADATA.txt')).write_text(dist.read_text('METADATA'))
(root/'runtime-source.json').write_text(json.dumps(records,indent=2)+'\n')
cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
with tarfile.open(root/'inductor-cache-at-stop.tar.gz','w:gz') as t:
    t.add(cache,arcname='inductor')
print('PASS runtime source identities and stopped compiler artifacts preserved')
