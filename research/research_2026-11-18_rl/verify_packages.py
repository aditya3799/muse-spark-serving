"""Verify candidate packages without modifying the shared SGLang checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from build_aux import BASE, DIGEST, HERE, ROOT

packages=[
    ROOT/"release_2026-11-18/candidate.diff",
    ROOT/"research_2026-11-18_rl_prefill/candidate.diff",
    ROOT/"research_2026-11-18_rl_prefill/target_dense/candidate.diff",
    HERE/"aux_candidate.diff",
    HERE/"decode_vocab/candidate.diff",
    HERE/"decode_combined/three_candidate.diff",
    ROOT/"research_2026-11-18_rl_prefill/decode_pages/candidate.diff",
    ROOT/"research_2026-11-18_lt_audit/integrated_candidate.diff",
    ROOT/"research_2026-11-18_rl_dflash/packed_taps_direct_kv.diff",
    ROOT/"research_2026-11-18_rl_experts/upstream_scale_order/candidate.diff",
    ROOT/"research_2026-11-18_rl_experts/combined/candidate.diff",
    ROOT/"research_2026-11-18_rl_experts/grouped_route/general_joint.diff",
    ROOT/"research_2026-11-18_rl_prefill/sequence_parallel/candidate.diff",
    ROOT/"research_2026-11-18_window_repair/early_candidate.diff",
]
checks=[]
with tempfile.TemporaryDirectory() as directory:
    for path in packages:
        raw=path.read_bytes()
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(directory)/"index"))
        def git(*args):
            return subprocess.check_output(["git","-C",str(ROOT/"sglang"),*args],env=env)
        git("read-tree",BASE)
        git("apply","--cached","--check",str(path))
        git("apply","--cached",str(path))
        files=git("diff","--cached","--name-only","--no-renames",BASE).decode().splitlines()
        assert all(name.startswith("python/sglang/") and not name.startswith("python/sglang/srt/model_loader/") for name in files)
        assert len(raw)<=400_000
        added=[line[1:] for line in raw.splitlines() if line.startswith(b"+") and not line.startswith(b"+++")]
        assert max(map(len,added))<=400
        digest=hashlib.sha256(raw).hexdigest()
        if path==packages[0]: assert digest==DIGEST
        checks.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest,bytes=len(raw),
                           files=len(files),max_added_line_bytes=max(map(len,added)),applies_to=BASE))
(HERE/"verified_packages.json").write_text(json.dumps(checks,indent=2)+"\n")
print("Verified",len(checks),"patches; maximum added line length",max(item["max_added_line_bytes"] for item in checks))