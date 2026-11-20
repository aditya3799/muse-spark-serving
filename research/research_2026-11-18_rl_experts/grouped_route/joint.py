"""Compose the disjoint 16K activation and verified grouped-route delta in an isolated index."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
here=Path(__file__).resolve().parent
root=here.parents[1];repo=root/"sglang"
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
ballot=root/"research_2026-11-18_rl_prefill/candidate.diff"
release=root/"release_2026-11-18/candidate.diff"
sha=lambda a:hashlib.sha256(a).hexdigest()
assert sha(ballot.read_bytes())=="33a0a65719afafe691e5c685a7a32ead3194112fc18ce8434f3b31880ce9a721"
name="python/sglang/kernels/ops/moe/route_activation.py"
with tempfile.TemporaryDirectory(dir=here) as tmp:
    objects=Path(tmp)/"objects";objects.mkdir()
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(tmp)/"index"),GIT_OBJECT_DIRECTORY=str(objects),GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/".git/objects"))
    def git(*args,data=None):
        return subprocess.run(["git","-C",str(repo),*args],input=data,env=env,capture_output=True,check=True).stdout
    def apply(p):git("apply","--cached","--whitespace=error","-",data=p)
    git("read-tree",base);apply(release.read_bytes());parent=git("write-tree").decode().strip()
    git("read-tree",base);apply(ballot.read_bytes());apply((here/"delta.diff").read_bytes())
    src=git("show",":"+name).decode()
    assert src.count("ids.shape[0] >= 512")==1
    src=src.replace("ids.shape[0] >= 512","ids.shape[0] == 16384")
    ast.parse(src);obj=git("hash-object","-w","--stdin",data=src.encode()).decode().strip()
    git("update-index","--cacheinfo","100644",obj,name)
    target=git("write-tree").decode().strip()
    for f,tree in (("joint_16k.diff",base),("joint_16k_delta.diff",parent)):
        patch=git("diff","--binary","--no-renames",tree,target)
        git("read-tree",tree);apply(patch);assert git("write-tree").decode().strip()==target
        (here/f).write_bytes(patch)
    kernel="python/sglang/kernels/jit/csrc/elementwise/route_activation.cuh"
    group="python/sglang/kernels/ops/moe/muse_spark_grouped_routes.py"
    assert sha(git("show",target+":"+kernel))==sha((root/"research_2026-11-18_rl_prefill/route_activation.cuh").read_bytes())
    assert sha(git("show",target+":"+group))==sha((here/"packaged_grouped.py").read_bytes().replace(b'\r\n',b'\n'))
    result=dict(sha256=sha((here/"joint_16k.diff").read_bytes()),base=base,parent=parent,target=target,
        group_sha=sha((here/"candidate.diff").read_bytes()),group8_input_sha=sha(ballot.read_bytes()),
        static_coverage={"grouped":"M128/768 SM86 when plan qualifies; only M128 qualified on archived SM90 plan","group8":"M16384 only; exact original native code and method; all other shapes invoke scored scale and SiLU"},
        evidence="Composition and source preservation only. No same-package balanced/skewed full-FFN measured benefit claimed.")
    (here/"joint_16k.json").write_text(json.dumps(result,indent=2))
    print(result["sha256"])
