"""Bounded public-source collector; no installation or execution of upstream code."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent / "sources"
HERE.mkdir(exist_ok=True)
LIMIT = 20_000_000
used = 0
manifest = []

def fetch(url):
    global used
    request = urllib.request.Request(url, headers={"User-Agent": "OpenKernelResearch-source-audit"})
    with urllib.request.urlopen(request, timeout=40) as response:
        data = response.read(3_000_001)
    if len(data) > 3_000_000 or used + len(data) > LIMIT:
        raise ValueError("source byte budget exceeded")
    used += len(data)
    return data

repos = {
    "cohere-ai/cohere-megakernel": ("main", ["src/", "CMakeLists.txt"]),
    "Tencent/hpc-ops": ("main", ["src/attention/", "hpc/attention", "tests/test_attention"]),
    "mirage-project/mirage": ("mpk", ["include/mirage/persistent_kernel", "python/mirage/", "demo/"]),
}
for repo, (branch, prefixes) in repos.items():
    dest = HERE / repo.replace("/", "__")
    dest.mkdir(exist_ok=True)
    try:
        head = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{branch}"))
        commit = head["sha"]
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        (dest / "tree.json").write_text(json.dumps(tree))
        selected = [row for row in tree["tree"] if row["type"] == "blob" and
                    row.get("size", 0) < 150_000 and any(row["path"].startswith(x) for x in prefixes)]
        for row in selected[:30]:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{row['path']}"
            data = fetch(url)
            target = dest / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest.append(dict(repo=repo, commit=commit, path=row["path"], url=url,
                                 bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        print(repo, commit, "selected", len(selected), "downloaded", min(30, len(selected)), flush=True)
    except Exception as error:
        manifest.append(dict(repo=repo, error=str(error)))
        print(repo, type(error).__name__, str(error), flush=True)
    (HERE / "manifest.json").write_text(json.dumps(dict(fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        byte_budget=LIMIT, bytes_downloaded=used, files=manifest), indent=2))
