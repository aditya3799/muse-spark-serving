"""Compose both local changes through a disposable Git index and object store."""
import ast
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[1]
repo = root/"sglang"
baseline = root/"release_2026-11-18/candidate.diff"
ballot = root/"research_2026-11-18_rl_prefill/candidate.diff"
ordering = here.parent/"upstream_scale_order"
digest = lambda x: hashlib.sha256(x).hexdigest()
inputs = {f.name+":"+f.parent.name:digest(f.read_bytes()) for f in (baseline,ballot,ordering/"candidate.diff")}
assert list(inputs.values()) == ["8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b", "33a0a65719afafe691e5c685a7a32ead3194112fc18ce8434f3b31880ce9a721","9ea3824c56c110645778bc54d3bea34cd8622a8c7d554f978c01e6a7502acbca"]
base = "297e54fd105e974cdf401e5b4d20253a22c16f90"
prefix = "python/sglang/"
sources = ["kernels/ops/moe/muse_spark_humming_split.py","srt/layers/muse_spark/muse_spark_nvfp4_moe.py"]
with tempfile.TemporaryDirectory(dir=here) as tmp:
    objects=Path(tmp)/"objects"; objects.mkdir()
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"),GIT_OBJECT_DIRECTORY=str(objects),
             GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/".git/objects"))
    def git(*args,data=None):
        return subprocess.run(["git","-C",str(repo),*args],input=data,capture_output=True,env=env,check=True).stdout
    def apply(patch):
        git("apply","--cached","--whitespace=error","-",data=patch)
    def idx(name,data):
        h=git("hash-object","-w","--stdin",data=data).decode().strip()
        git("update-index","--add","--cacheinfo","100644",h,name)
    git("read-tree",base);apply(baseline.read_bytes());parent=git("write-tree").decode().strip()
    git("read-tree",base);apply(ballot.read_bytes());ballot_tree=git("write-tree").decode().strip()
    for rel in sources:
        name=prefix+rel
        current=git("show",":"+name).decode()
        lines=current.splitlines(True); count=0
        for node in ast.parse(current).body:
            if not isinstance(node,ast.FunctionDef): continue
            chunk=''.join(lines[node.lineno-1:node.end_lineno])
            if 'HummingMethod.forward_layer(' not in chunk: continue
            for i in range(node.lineno-1,node.end_lineno):
                count+=lines[i].count("HummingMethod.forward_layer(")
                lines[i]=lines[i].replace("from humming.layer import HummingMethod","from sglang.kernels.ops.moe.scale_order import scale_order_forward").replace("HummingMethod.forward_layer(","scale_order_forward(")
        assert count==(4 if rel==sources[0] else 2)
        data=''.join(lines).encode();ast.parse(data);idx(name,data)
    name=prefix+"kernels/ops/moe/scale_order.py"
    idx(name,(ordering/"scale_order.py").read_bytes())
    target=git("write-tree").decode().strip()
    for label,tree in (("parent",parent),("ballot",ballot_tree),("combined",target)):
        for rel in sources:
            path=here/label/Path(rel).name;path.parent.mkdir(exist_ok=True)
            path.write_bytes(git("show",tree+":"+prefix+rel))
    for rel in ("kernels/ops/moe/route_activation.py","kernels/jit/csrc/elementwise/route_activation.cuh","kernels/ops/moe/scale_order.py"):
        (here/"combined"/Path(rel).name).write_bytes(git("show",target+":"+prefix+rel))
    for name,start in (("candidate.diff",base),("delta.diff",parent)):
        patch=git("diff","--binary","--no-renames",start,target)
        git("read-tree",start);apply(patch);assert git("write-tree").decode().strip()==target
        (here/name).write_bytes(patch)
    receipt=dict(inputs=inputs,sglang=base,source_trees={"parent":parent,"ballot":ballot_tree,"combined":target},candidate_sha256=digest((here/"candidate.diff").read_bytes()))
    (here/"composition.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt)


