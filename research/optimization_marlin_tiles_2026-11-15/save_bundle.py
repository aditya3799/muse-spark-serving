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
status = json.loads((destination / "H200_STATUS.json").read_text())
active_manifest_path = serving / "candidate_manifest.json"
active = json.loads(active_manifest_path.read_text())
assert active["unscored_job"] == status["job"]
active["last_observed_job_status"] = status["status"]
active["job_status_observed_at_utc"] = status["observed_at_utc"]
active["latest_profiling_review"] = "PROFILING_SESSION_2026-11-15.txt"
active_manifest_path.write_text(json.dumps(active, indent=2) + "\n", encoding="utf-8")
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
