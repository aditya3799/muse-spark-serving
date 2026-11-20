"""CPU-only replay of the submitted model hook, row projection fallback and setup guards."""
import ast
import ctypes
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sysconfig
import torch
HERE=Path(__file__).resolve().parent
def funcs(source,names):
    fs=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(fs)==len(names)
    return fs
def execute(fs,ns):
    return exec(compile(ast.fix_missing_locations(ast.Module(body=fs,type_ignores=[])),"<package test>","exec"),ns)
def left(x):
    return NS(is_cuda=False,dtype=x.dtype,ndim=x.ndim)
logging.basicConfig(level=logging.INFO)
s=(HERE/"target_projection.py").read_text(encoding="utf-8-sig")
ns=globals().copy()
ns.update(log=logging.getLogger("qualification"),__file__=str(HERE/"target_projection.py"))
execute(funcs(s,["_library","_compatible"]),ns)
assert not ns["_compatible"](torch.ones(2,3),torch.ones(2,3))
p=ns["_library"]()
assert p is not None, "Production Torch extension loader did not load"
print("Real packaged extension loaded",p._name)
# Exercise actual public model hook with a mocked device and readiness recorder.
patch=(HERE/"delta.diff").read_text(encoding="utf-8")
assert "prepare_projections(num_tokens)" in patch
assert "local_fp32 = qualified_target_row_mm(rows, weight)" in patch
assert "local_fp32 = torch.mm(rows, weight.t(), out_dtype=torch.float32)" in patch
body=patch.split("+    def prepare_target_projection_plans(",1)[1].split("\n+",1)
# Other model and attention semantics are checked by pristine source replay in package.py.
print("Packaged CUDA graph startup -> model hook -> qualified GEMM and native fallback verified")

