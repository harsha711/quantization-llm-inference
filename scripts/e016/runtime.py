#!/usr/bin/env python3
"""Archive and assert capture-relevant installed source, without model execution."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch.cuda.graphs
from transformers.cache_utils import StaticLayer, CacheLayerMixin
import transformers.cache_utils
import transformers.models.qwen2.modeling_qwen2
import transformers.integrations.sdpa_attention

root = Path(sys.argv[1])/'runtime-source'
root.mkdir()
modules = (torch.cuda.graphs, transformers.cache_utils, transformers.models.qwen2.modeling_qwen2,
           transformers.integrations.sdpa_attention)
records = {}
for module in modules:
    source = Path(inspect.getfile(module))
    data = source.read_bytes()
    (root/(module.__name__+'.py')).write_bytes(data)
    records[module.__name__] = dict(path=str(source), sha256=hashlib.sha256(data).hexdigest())
update = inspect.getsource(StaticLayer.update)
reset = inspect.getsource(CacheLayerMixin.reset)
assert 'index_copy_(2, cache_position' in update and 'cumulative_length' not in update
assert 'self.keys.zero_()' in reset and 'self.values.zero_()' in reset
(root/'identity.json').write_text(json.dumps(records, indent=2)+'\n')
print('PASS installed StaticLayer device-indexed writes, in-place reset, source hashes', records)
