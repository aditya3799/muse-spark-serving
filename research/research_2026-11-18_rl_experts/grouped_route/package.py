"""Assemble cumulative patch through a disposable index on the exact scored parent."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
here=Path(__file__).resolve().parent
root=here.parents[1]
repo=root/"sglang"
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
parent_patch=root/"release_2026-11-18/candidate.diff"
sha=lambda data:hashlib.sha256(data).hexdigest()
assert sha(parent_patch.read_bytes())=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
hoppers=json.loads((here/"hopper_groups.json").read_text())
parents=json.loads((here/"hopper_parent.json").read_text())
assert len(hoppers)==len(parents)==8
rows=set()
for row,ref in zip(hoppers,parents):
    if row["allowed"]:
        rows.add(((9,0),row["stage"],tuple(row["block"]),tuple(row["warp"]),row["stages"]))
    else:
        assert row["resources"]["STACK"]>ref["resources"]["STACK"]
gpu=json.loads((here/"fused_pilot.json").read_text())
for row in gpu["dispatch"]:
    rows.add(((8,6),row["stage"],tuple(row["block"]),tuple(row["warp"]),3))
gate="PROVEN_GEOMETRY = frozenset([\n"+"".join("    "+repr(r)+",\n" for r in sorted(rows))+"])\n"
module=(here/"grouped_fused.py").read_text().rstrip()+"\n\n"+gate+"\n"+(here/"production_adapter.py").read_text().rstrip()+"\n"
ast.parse(module)
module_name="python/sglang/kernels/ops/moe/muse_spark_grouped_routes.py"
entry_name="python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"
with tempfile.TemporaryDirectory(dir=here) as td:
    tmp=Path(td);objects=tmp/"objects";objects.mkdir()
    env=dict(os.environ,GIT_INDEX_FILE=str(tmp/"index"),GIT_OBJECT_DIRECTORY=str(objects),GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/".git/objects"))
    def git(*args,data=None):
        return subprocess.run(["git","-C",str(repo),*args],input=data,capture_output=True,env=env,check=True).stdout
    def put(name,data):
        obj=git("hash-object","-w","--stdin",data=data).decode().strip()
        git("update-index","--add","--cacheinfo","100644",obj,name)
    def apply(data):git("apply","--cached","--whitespace=error","-",data=data)
    git("read-tree",base);apply(parent_patch.read_bytes());parent=git("write-tree").decode().strip()
    entry=git("show",":"+entry_name).decode()
    needle="        defer_downscale = self.use_humming and self._post_expert_norm_weight is not None\n"
    assert entry.count(needle)==1
    entry=entry.replace(needle,needle+"""        if defer_downscale and hidden_states.shape[0] in (128, 768):
            from sglang.kernels.ops.moe.muse_spark_grouped_routes import try_grouped

            result = try_grouped(self, hidden_states, topk_ids, topk_scores)
            if result is not None:
                return result
""")
    ast.parse(entry);put(entry_name,entry.encode());put(module_name,module.encode());target=git("write-tree").decode().strip()
    for name,start in (("candidate.diff",base),("delta.diff",parent)):
        p=git("diff","--binary","--no-renames",start,target)
        git("read-tree",start);apply(p)
        assert git("write-tree").decode().strip()==target
        (here/name).write_bytes(p)
    (here/"packaged_grouped.py").write_text(module)
    (here/"packaged_owner.py").write_text(entry)
    (here/"package.json").write_text(json.dumps(dict(base=base,scored_parent=parent,target=target,candidate_sha256=sha((here/"candidate.diff").read_bytes()),tested_inner_sha256=sha((here/"grouped_fused.py").read_bytes()),geometries=list(map(str,sorted(rows))),sm90_m768_guard="Heavy W13/W2 stack 376/368 bytes versus 304 bytes both indexed. Shape fallback is all or none on exact plan.",sm90_runtime="unmeasured"),indent=2))
print(sha((here/"candidate.diff").read_bytes()))
