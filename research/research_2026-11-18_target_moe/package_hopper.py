"""Create and verify a source-only patch against the pinned SGLang tree in an isolated index."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from build_hopper import FLAGS,materialize,replace
from specialize import serving_source

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
BASE = "297e54fd105e974cdf401e5b4d20253a22c16f90"
PREFIX = "python/sglang/srt/layers/muse_spark/"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    out = p.parse_args().out.resolve()
    out.mkdir(exist_ok=True,parents=True)
    release = ROOT/"release_2026-11-18"
    parent = (release/"candidate.diff").read_bytes()
    assert hashlib.sha256(parent).hexdigest()=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    with tempfile.TemporaryDirectory() as tmp:
        objects = Path(tmp)/"objects"
        objects.mkdir()
        repo = ROOT/"sglang"
        odb = subprocess.check_output(["git","-C",str(repo),"rev-parse","--path-format=absolute","--git-path","objects"]).decode().strip()
        env = dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"idx"),GIT_OBJECT_DIRECTORY=str(objects),GIT_ALTERNATE_OBJECT_DIRECTORIES=odb)
        def git(*args,data=None):
            return subprocess.run(["git","-C",str(repo),*args],input=data,env=env,capture_output=True,check=True).stdout
        git("read-tree",BASE)
        git("apply","--cached","-",data=parent)
        name = PREFIX+"muse_spark_nvfp4_moe.py"
        adapter = git("show",":"+name).decode()
        adapter = replace(adapter,"        layer._humming_layer = _prepare_humming_experts(raw, torch)",
            "        from .hopper_backend import HopperExperts\n        packed = HopperExperts.prepare(layer, raw)\n"
            "        layer._humming_layer = packed if packed is not None else _prepare_humming_experts(raw, torch)")
        adapter = replace(adapter,"    from sglang.kernels.ops.moe.muse_spark_humming_split import split_expert_outputs",
            "    from .hopper_backend import HopperExperts\n    if isinstance(layer, HopperExperts):\n"
            "        return layer.project(x, local_ids, defer_downscale)\n"
            "    from sglang.kernels.ops.moe.muse_spark_humming_split import split_expert_outputs")
        files = {"muse_spark_nvfp4_moe.py":adapter,"hopper_expert.cu":serving_source(materialize(),FLAGS),
            **{n:(HERE/n).read_text() for n in ("hopper_pack.py","hopper_backend.py")}}
        for short,body in files.items():
            if short.endswith(".py"): ast.parse(body)
            data = ("\n".join(line.rstrip() for line in body.splitlines()).rstrip()+"\n").encode()
            (out/short).write_bytes(data)
            oid = git("hash-object","-w","--stdin",data=data).decode().strip()
            git("update-index","--add","--cacheinfo","100644",oid,PREFIX+short)
        patch = git("diff","--cached","--no-ext-diff","--no-color",BASE)
        assert len(patch)<=400000
        assert max(len(x)-1 for x in patch.splitlines() if x.startswith(b"+") and not x.startswith(b"+++"))<=400
        git("read-tree",BASE)
        git("apply","--cached","--check","--whitespace=error","-",data=patch)
        (out/"candidate.diff").write_bytes(patch)
        (out/"results.json").write_text(json.dumps(dict(base=BASE,sha256=hashlib.sha256(patch).hexdigest(),
            bytes=len(patch),applies=True),indent=2))

if __name__=="__main__":
    main()

