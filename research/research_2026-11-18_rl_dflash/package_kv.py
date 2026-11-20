"""Confirm parent+delta and joint full patch on a fresh git index."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT / "sglang"
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PARENT = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"


def main():
    native = (ROOT / "release_2026-11-18/candidate.diff").read_bytes()
    assert hashlib.sha256(native).hexdigest() == PARENT
    # The packed-tap baseline is already a complete proven-parent patch.
    packed = (ROOT / "research_2026-11-18_rl/aux_candidate.diff").read_bytes()
    delta = (HERE / "kv_trial_parent_delta.diff").read_text(encoding="utf8")
    chunks = delta.split("--- a/")
    decorated = []
    for part in chunks[1:]:
        name = part.splitlines()[0]
        decorated.append(f"diff --git a/{name} b/{name}\n--- a/" + part)
    addition = "".join(decorated).encode()
    receipts = {}
    for label, baseline in (
        ("direct_kv_parent", native),
        ("packed_taps_direct_kv", packed),
    ):
        out = HERE / (label + ".diff")
        out.write_bytes(baseline.rstrip(b"\n") + b"\n" + addition)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp) / "index"))

            def git(args):
                return subprocess.run(
                    ["git", "-C", str(REPO), *args],
                    env=env,
                    check=True,
                    capture_output=True,
                )

            git(["read-tree", BASE])
            git(["apply", "--cached", "--check", "--whitespace=error", str(out)])
            git(["apply", "--cached", str(out)])
            names = (
                git(["diff", "--cached", "--name-only", BASE])
                .stdout.decode()
                .splitlines()
            )
            assert all(
                p.startswith("python/sglang/") and "/model_loader/" not in p
                for p in names
            )
        assert out.stat().st_size < 400000
        receipts[label] = dict(
            path=out.name,
            sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            bytes=out.stat().st_size,
            files=names,
            apply_cached_check=True,
        )
    print(json.dumps({k: (v["sha256"], v["bytes"]) for k, v in receipts.items()}))
    (HERE / "full_patch_receipts.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
