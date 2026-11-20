"""Only exact FA3 dependency headers; stream archive without retaining archive."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PIN = '57e3cfb47a2d9e0d46eb6335c3dc411498efa198'
ROOT = HERE / 'sources/cutlass_fa3'
url = 'https://codeload.github.com/NVIDIA/cutlass/tar.gz/' + PIN
total = 0
records = []
start = time.monotonic()
with urllib.request.urlopen(url, timeout=60) as response:
    with tarfile.open(fileobj=response, mode='r|gz') as archive:
        for entry in archive:
            pieces = PurePosixPath(entry.name).parts
            if len(pieces)<3 or pieces[1]!='include' or not entry.isfile():
                continue
            rel = PurePosixPath(*pieces[1:])
            if '..' in rel.parts:
                raise RuntimeError('unsafe archive name')
            total += entry.size
            if total > 64*1024*1024:
                raise RuntimeError('bounded source budget exceeded')
            data = archive.extractfile(entry).read()
            assert len(data) == entry.size
            path = ROOT / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append(dict(path=str(rel), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
manifest = dict(pin=PIN, url=url, bytes=total, elapsed_s=time.monotonic()-start, files=records)
(ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(pin=PIN, headers=len(records), bytes=total, elapsed_s=manifest['elapsed_s'])))
