#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
py=/home/aditya/.venvs/okc-marlin-tiles/bin/python
cu=$("$py" -c 'import pathlib,sysconfig;print(pathlib.Path(sysconfig.get_paths()["purelib"])/"nvidia/cu13")')
g++ -O2 -std=c++17 -fPIC -shared sweep.cc -I"$cu/include" -I"$cu/include/cccl" \
    -L"$cu/lib" -Wl,-rpath,"$cu/lib" -l:libcublasLt.so.13 -o sweep.so

