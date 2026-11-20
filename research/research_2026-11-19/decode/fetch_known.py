"""Fetch known public files at advertised Git revisions; bounded to 3 MB/file, 20 MB total."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent / "sources"
files = {
    "cohere-ai/cohere-megakernel": ["src/decode/megakernel.cuh", "src/decode/schedule.py", "src/decode/launch.cuh", "src/decode/gemm-n8-wgmma.cuh"],
    "Tencent/hpc-ops": ["src/attention/decode/decode.h", "src/attention/decode/sched_task_info.h", "src/attention/entry.cc"],
    "mirage-project/mirage": ["include/mirage/persistent_kernel/persistent_kernel.cuh", "include/mirage/persistent_kernel/runtime_header.h"],
}
records = []
total = 0
for repo, paths in files.items():
    try:
        line = subprocess.check_output(["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"], timeout=35, text=True)
        sha = line.split()[0]
        assert len(sha) == 40
        for path in paths:
            target = HERE / repo.replace("/", "__") / path
            url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
            with urllib.request.urlopen(url, timeout=35) as response:
                data = response.read(3_000_001)
            assert len(data) <= 3_000_000 and total + len(data) <= 20_000_000
            total += len(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records.append(dict(repo=repo, path=path, commit=sha, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        print(repo, sha, "files", len(paths), flush=True)
    except Exception as error:
        records.append(dict(repo=repo, error=str(error)))
        print(repo, str(error), flush=True)
    (HERE / "known_manifest.json").write_text(json.dumps(dict(records=records, total_bytes=total,
        retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=2))
