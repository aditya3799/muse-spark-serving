"""Finish the reviewable experiment bundle, without changing serving sources."""

import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from marlin_bench import EXPECTED_SHA, SOURCE

root = Path(__file__).resolve().parent
serving = Path(r"D:\OpenKernelChallenge\serving")
source = serving / "sglang" / SOURCE
assert hashlib.sha256(source.read_bytes()).hexdigest() == EXPECTED_SHA
original = source.read_text(encoding="utf-8")
old = "if num_tokens * topk / self.num_local_experts / block_size_m < 0.9:"
new = "if num_tokens * topk / self.global_num_experts / block_size_m < 0.9:"
assert original.count(old) == 1
candidate = original.replace(old, new)
delta = "".join(
    difflib.unified_diff(
        original.splitlines(True),
        candidate.splitlines(True),
        fromfile="a/" + SOURCE,
        tofile="b/" + SOURCE,
    )
)
(root / "EP_DENSITY_DELTA.diff").write_text(delta, encoding="utf-8")
test = subprocess.run(
    [sys.executable, "-m", "unittest", "-v", "test_experiment"],
    cwd=root,
    text=True,
    capture_output=True,
)
(root / "unit_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
if test.returncode:
    raise RuntimeError("Unit checks failed")
report = {
    "phase": "isolated Marlin EP tile experiment 01",
    "status": "structural checks passed; real-kernel benchmark execution pending",
    "baseline_source_sha256": EXPECTED_SHA,
    "submitted_patch_sha256": hashlib.sha256(
        (serving / "patch.diff").read_bytes()
    ).hexdigest(),
    "unit_tests_passed": 12,
    "structure": json.loads(
        (root / "local_structure/structure_summary.json").read_text()
    ),
    "cuda_route_checks": json.loads(
        (root / "local_structure/cuda_route_checks.json").read_text()
    ),
    "full_gpu_benchmark_run": False,
    "active_source_modified": False,
    "new_submission": False,
    "delta_scope": "one-line experiment relative to the submitted source; not a cumulative submission patch",
}
(root / "experiment_manifest.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
files = [
    p
    for p in root.rglob("*")
    if p.is_file()
    and "__pycache__" not in p.parts
    and ".ruff_cache" not in p.parts
    and p.name != "bundle_checksums.json"
]
checksums = {
    str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(
        p.read_bytes()
    ).hexdigest()
    for p in files
}
(root / "bundle_checksums.json").write_text(
    json.dumps(checksums, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
