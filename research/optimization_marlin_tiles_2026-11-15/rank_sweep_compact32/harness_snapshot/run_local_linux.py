"""Launch a component smoke test using the experiment's isolated Linux venv."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--entry",
        choices=(
            "kernel_runtime.py",
            "validate_numeric_reference.py",
            "profile_marlin.py",
            "diagnose_replay.py",
            "validate_adaptive_selector.py",
            "validate_exported_candidate.py",
            "measure_memory.py",
            "rank_sweep.py",
        ),
        default="kernel_runtime.py",
    )
    args, rest = parser.parse_known_args()
    if platform.system() != "Linux" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the dedicated Linux virtual environment")
    args.out.mkdir(parents=True, exist_ok=True)
    compiler_dist = metadata.distribution("nvidia-cuda-nvcc")
    candidates = [
        Path(compiler_dist.locate_file(f)).resolve()
        for f in compiler_dist.files
        if str(f).endswith("/bin/nvcc")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Could not identify nvcc: {candidates}")
    cuda_home = candidates[0].parent.parent
    # NVIDIA wheels use lib/ whereas this SGLang toolchain expects lib64/.
    # Keep the compatibility link within this dedicated environment only.
    if not (cuda_home / "lib64").exists() and (cuda_home / "lib").is_dir():
        (cuda_home / "lib64").symlink_to(cuda_home / "lib", target_is_directory=True)
    cudart_link = cuda_home / "lib64/libcudart.so"
    if not cudart_link.exists():
        libraries = sorted((cuda_home / "lib64").glob("libcudart.so.*"))
        if not libraries:
            raise RuntimeError("CUDA runtime wheel does not contain libcudart")
        cudart_link.symlink_to(libraries[0].name)
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).parent), str(cuda_home / "bin"), env.get("PATH", ""))
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(cuda_home / "lib64"), "/usr/lib/wsl/lib", env.get("LD_LIBRARY_PATH", ""))
    )
    env["SGLANG_CACHE_DIR"] = str(Path.home() / ".cache/okc-marlin-tiles")
    env["SGLANG_JIT_CACHE_DIR"] = str(Path.home() / ".cache/okc-marlin-tiles/jit")
    env["MAX_JOBS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    headers = Path(sys.prefix) / "python-headers/usr/include"
    if (headers / "python3.12/Python.h").is_file():
        env["CPATH"] = os.pathsep.join(
            (str(headers / "python3.12"), str(headers), env.get("CPATH", ""))
        )
    versions = {
        dist.metadata["Name"]: dist.version for dist in metadata.distributions()
    }
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": versions,
        "cuda_home": str(cuda_home),
        "compiler_version": subprocess.check_output(
            [str(candidates[0]), "--version"], text=True
        ),
        "host_free_bytes": shutil.disk_usage("/mnt/c").free,
        "component_only": True,
        "max_jobs": 2,
    }
    (args.out / "environment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "packages"}, indent=2),
        flush=True,
    )
    if not args.inspect_only:
        snapshot = args.out / "harness_snapshot"
        snapshot.mkdir(exist_ok=False)
        hashes = {}
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source, snapshot / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (args.out / "harness_sha256.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8"
        )
        command = [
            sys.executable,
            str(snapshot / args.entry),
            "--sglang-root",
            str(args.sglang_root),
            "--out",
            str(args.out),
        ] + rest
        os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
