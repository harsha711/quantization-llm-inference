"""Read-only archival with byte-identical-file deduplication, preserving every path."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert a.root.is_absolute() and a.output.is_absolute() and not a.output.exists()
roots=['results/e017_20261109','scripts/e017','scripts/e014','scripts/e004','scripts/e005','src','include','tests']
files=[]
for name in roots:
    files.extend((f,f.relative_to(a.root).as_posix()) for f in (a.root/name).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
for name in ['CMakeLists.txt','scripts/build.sh','scripts/check_inblock.sh','build/CMakeCache.txt','AGENTS.md','README.md','docs/CONTRACT.md','docs/EXPERIMENTS.md']:
    if (a.root/name).is_file():files.append((a.root/name,name))
for path,name in [('/home/hpuvvad/csc548-w4a16-e006-20261106/results/e006_20261106/patched/libe005.so','executed-binaries/libe005.so'),('/home/hpuvvad/csc548-w4a16-e004-20261106/results/e004_20261106/libe004.so','executed-binaries/libe004.so')]:
    files.append((Path(path),name))
seen={};manifest=[];unique=0
with tarfile.open(a.output,'w:gz',compresslevel=1,dereference=True) as archive:
    for source,name in sorted(files,key=lambda item:item[1]):
        before=source.stat();digest=hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        key=(before.st_size,digest.hexdigest())
        info=archive.gettarinfo(str(source),arcname=name)
        if key in seen:
            info.type=tarfile.LNKTYPE;info.linkname=seen[key];info.size=0;archive.addfile(info)
        else:
            with source.open('rb') as stream:archive.addfile(info,stream)
            seen[key]=name;unique+=before.st_size
        after=source.stat();assert (before.st_mtime_ns,before.st_size)==(after.st_mtime_ns,after.st_size),source
        manifest.append(f'{digest.hexdigest()}  {name}\n')
    # Archive-specific manifest is embedded, excludes itself and output archive.
    import io
    data=''.join(manifest).encode();info=tarfile.TarInfo('retrieval.sha256');info.size=len(data);archive.addfile(info,io.BytesIO(data))
print(json.dumps(dict(files=len(files),unique_bytes=unique,archive_bytes=a.output.stat().st_size,archive=str(a.output))),flush=True)
