"""Resolve a source-built local cuBLASLt helper once per installation."""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile

import torch


@lru_cache(maxsize=1)
def native_binary():
    if os.name != "posix":
        raise OSError("cuBLASLt runtime builder requires POSIX file locking")
    import fcntl

    root = Path(__file__).resolve().parent
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    candidates = [nvidia / "cu13", nvidia / "cublas", Path(os.getenv("CUDA_HOME", "/usr/local/cuda"))]
    prefix = next((p for p in candidates if (p / "include/cublasLt.h").is_file()), None)
    if prefix is None:
        raise OSError("cuBLASLt development headers unavailable")
    libraries = [item for libdir in ("lib", "lib64", "targets/x86_64-linux/lib")
                 for item in sorted((prefix / libdir).glob("libcublasLt.so*"))]
    compiler = shutil.which(os.getenv("CXX", "g++"))
    if not libraries or not compiler:
        raise OSError("cuBLASLt shared library or C++ compiler unavailable")
    library = libraries[0].resolve()
    src = root / "vendor.cc"
    digest = hashlib.sha256(src.read_bytes())
    for value in (torch.__version__, torch.version.cuda, str(library), str(library.stat().st_mtime_ns), compiler):
        digest.update(str(value).encode())
    cached = Path(tempfile.gettempdir()) / f"sglang-dflash-projector-{os.getuid()}" / digest.hexdigest()
    cached.mkdir(parents=True, mode=0o700, exist_ok=True)
    output = cached / "sweep.so"
    with (cached / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            tmp = cached / f"sweep.{os.getpid()}.so"
            command = [compiler, "-O2", "-std=c++17", "-shared", "-fPIC"]
            includes = [prefix / "include", nvidia / "cuda_runtime/include", prefix / "include/cccl"]
            for inc in includes:
                if inc.is_dir():
                    command.extend(("-I", str(inc)))
            command.extend((str(src), str(library), "-Wl,-rpath," + str(library.parent), "-o", str(tmp)))
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=90)
                os.replace(tmp, output)
            except (subprocess.SubprocessError, OSError) as exc:
                raise OSError("cuBLASLt projection helper build failed") from exc
            finally:
                tmp.unlink(missing_ok=True)
    return output

