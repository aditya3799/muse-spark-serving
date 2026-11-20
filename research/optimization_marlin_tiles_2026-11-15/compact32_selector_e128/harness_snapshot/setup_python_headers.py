"""Extract matching Ubuntu Python headers into the experiment venv, not /usr."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

prefix = Path(sys.prefix).resolve()
if prefix.name != "okc-marlin-tiles" or sys.version_info[:2] != (3, 12):
    raise RuntimeError("Expected the dedicated Python 3.12 experiment environment")
downloads = prefix / "header-downloads"
downloads.mkdir(exist_ok=True)
version = "3.12.3-1ubuntu0.17"
subprocess.run(
    ["apt-get", "download", f"libpython3.12-dev={version}"], cwd=downloads, check=True
)
packages = list(downloads.glob(f"libpython3.12-dev_{version}_amd64.deb"))
if len(packages) != 1:
    raise RuntimeError("Expected one downloaded header package")
destination = prefix / "python-headers"
destination.mkdir(exist_ok=True)
subprocess.run(["dpkg-deb", "-x", str(packages[0]), str(destination)], check=True)
assert (destination / "usr/include/python3.12/Python.h").is_file()
report = {
    "package": packages[0].name,
    "sha256": hashlib.sha256(packages[0].read_bytes()).hexdigest(),
    "destination": str(destination),
    "system_packages_changed": False,
}
(destination / "manifest.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
