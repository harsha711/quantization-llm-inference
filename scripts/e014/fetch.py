#!/usr/bin/env python3
"""Download only the declared checkpoint on allocated compute and hash inputs."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from huggingface_hub import snapshot_download
import torch

MODEL = 'Qwen/Qwen2.5-1.5B'
REVISION = '8faed761d45a263340a0528343f099c05c9a4323'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


if __name__ == '__main__':
    root = Path(sys.argv[1])
    assert os.environ['OMPI_COMM_WORLD_SIZE'] == '1'
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 5060 Ti'
    assert torch.cuda.get_device_capability() == (12,0)
    assert str(torch.__version__).startswith('2.14.0')
    torch.ones(16,device='cuda').sum().item()
    start = time.perf_counter()
    model = Path(snapshot_download(MODEL,revision=REVISION,
        allow_patterns=['*.json','*.safetensors','*.txt','LICENSE','README.md']))
    assert model.name == REVISION
    files = {p.name: dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(model.iterdir()) if p.is_file()}
    assert 'model.safetensors' in files and 'tokenizer.json' in files
    record = dict(model_id=MODEL,revision=REVISION,path=str(model),files=files,
        download_s=time.perf_counter()-start,torch=torch.__version__,cuda=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in
            ('torch','transformers','tokenizers','huggingface-hub','safetensors','numpy')})
    (root/'model.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
