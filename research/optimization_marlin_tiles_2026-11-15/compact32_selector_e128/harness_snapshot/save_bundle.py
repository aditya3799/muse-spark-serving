"""Copy only checksum-listed experiment artifacts into the authorized project."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
serving = Path(r"D:\OpenKernelChallenge\serving").resolve()
destination = (serving / "optimization_marlin_tiles_2026-11-15").resolve()
assert destination.is_relative_to(serving)
checksums = json.loads((root / "bundle_checksums.json").read_text())
for relative, expected in checksums.items():
    source = (root / relative).resolve()
    target = (destination / relative).resolve()
    assert source.is_relative_to(root) and target.is_relative_to(destination)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected, relative
shutil.copy2(root / "bundle_checksums.json", destination / "bundle_checksums.json")
shutil.copy2(
    destination / "NEXT_OPTIMIZATION.updated.txt", serving / "NEXT_OPTIMIZATION.txt"
)
shutil.copy2(
    destination / "SESSION_REVIEW.txt", serving / "PROFILING_SESSION_2026-11-15.txt"
)
print(
    json.dumps(
        dict(
            status="saved_and_hash_verified",
            files=len(checksums),
            destination=str(destination),
        ),
        indent=2,
    )
)
