"""Fetch only quoted local header dependencies at the already audited FA3 pin."""
from collections import deque
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import urllib.request

HERE = Path(__file__).resolve().parent
PIN = 'f89bc2306632d1ec5f97b014dded4254f5b4a907'
ROOT = HERE / 'sources/fa3'
prior = HERE.parents[1] / 'research_2026-11-19/decode/sources/sgl-flash-attn'
queue = deque(['hopper/flash_fwd_launch_template.h'])
seen = set()
records = []
downloaded = 0
while queue:
    name = queue.popleft()
    if name in seen:
        continue
    seen.add(name)
    path = ROOT / name
    source = prior / PurePosixPath(name).name
    own = HERE / 'sources' / PurePosixPath(name).name
    if path.exists():
        data = path.read_bytes()
        origin = 'existing own snapshot'
    elif source.exists() and name.startswith('hopper/'):
        data = source.read_bytes()
        origin = 'shared pinned decode snapshot'
    elif own.exists() and name.startswith('hopper/'):
        data = own.read_bytes()
        origin = 'own pinned snapshot'
    else:
        url = f'https://raw.githubusercontent.com/sgl-project/sgl-flash-attn/{PIN}/{name}'
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(2*1024*1024)
        downloaded += len(data)
        if downloaded > 4*1024*1024:
            raise RuntimeError('bounded header-download budget exceeded')
        origin = 'downloaded pinned source'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                        url=f'https://github.com/sgl-project/sgl-flash-attn/blob/{PIN}/{name}', origin=origin))
    text = data.decode()
    for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
        if include.startswith(('cute/', 'cutlass/', 'ATen/', 'c10/', 'torch/')):
            continue
        if include in ('cuda_runtime.h', 'cuda_runtime_api.h', 'cuda.h'):
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), include))
        if target.startswith('../'):
            raise RuntimeError('unexpected include outside repo: '+target)
        queue.append(target)
(HERE/'sources/fa3_manifest.json').write_text(json.dumps(dict(pin=PIN, downloaded_bytes=downloaded,
                                                            files=records), indent=2))
print(json.dumps(dict(headers=len(records), downloaded_bytes=downloaded)))
