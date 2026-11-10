"""Extract only the pinned official packing operator, without installing vLLM."""
import hashlib
from pathlib import Path
import zipfile
archive=Path('external/vllm-0.33.0-cp38-abi3-manylinux_2_28_x86_64.whl')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='4b6bdec4ab269990bf70d45a3c1e94c3760a27064513760193b26d6f0da883aa'
with zipfile.ZipFile(archive) as source:
    data=source.read('vllm/_C_stable_libtorch.abi3.so')
assert hashlib.sha256(data).hexdigest()=='e634b80e72078e964f8bd2ac541d66f2a900920773cf993544609612190bd9a9'
with Path('external/_C_stable_libtorch.abi3.so').open('xb') as output:output.write(data)
print('PASS exact official packer payload')
