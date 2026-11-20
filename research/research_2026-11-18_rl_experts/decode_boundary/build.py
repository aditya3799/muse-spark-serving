"""Build a small incremental patch and check the intended dispatch on CPU."""
from __future__ import annotations
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import types

here=Path(__file__).resolve().parent
root=here.parents[1]
repo=root/"sglang"
base="297e54fd105e974cdf401e5b4d20253a22c16f90"
sha=lambda b:hashlib.sha256(b).hexdigest()
patch=root/"release_2026-11-18/candidate.diff"
judged=json.loads((root/"release_2026-11-18/scored_submission_status.json").read_text())
assert sha(patch.read_bytes())==judged["result"]["patch"]["sha256"]
model="python/sglang/srt/models/muse_spark.py"
gather="python/sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py"
norm="python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py"
def get(path):
    p=root/"release_2026-11-18/candidate_sources"/path
    if p.exists():return p.read_text()
    return subprocess.check_output(["git","-C",str(repo),"show",f"{base}:{path}"]).decode()
old=get(gather)
new=old.replace("    norm_residual_launch_config,\n","    _launch_config,\n    norm_residual_launch_config,\n")
anchor="def _supports_clusters() -> bool:\n    return torch.cuda.get_device_capability()[0] >= 10"
new=new.replace(anchor,'''def _device_major() -> int:
    return torch.cuda.get_device_capability()[0]


def _boundary_config(hidden_size: int, num_tokens: int) -> tuple[int, int, bool]:
    if _device_major() == 9:
        # Match the standalone Hopper JIT norm's one-CTA arithmetic and mapping.
        return 1, *_launch_config(hidden_size, num_tokens)
    return norm_residual_launch_config(hidden_size=hidden_size, num_tokens=num_tokens)''')
new=new.replace("if not _supports_clusters():","if _device_major() < 9:")
new=new.replace("cluster_size, _, _ = norm_residual_launch_config(","cluster_size, _, _ = _boundary_config(")
new=new.replace("config = norm_residual_launch_config(hidden_size=hidden_size, num_tokens=num_tokens)","config = _boundary_config(hidden_size=hidden_size, num_tokens=num_tokens)")
assert new.count("_boundary_config(")==3 and "_supports_clusters" not in new
mo=get(model)
mn=mo.replace("    small_projection,\n","    small_projection,\n    small_projection_local,\n")
anchor="            if self._can_shard_small_projection(output, self.post_expert_proj.weight):\n"
assert mo.count(anchor)==1
mn=mn.replace(anchor,anchor+"                if self._defers_unpool_gather(comm, output.shape[0]):\n                    return small_projection_local(comm, output, self.post_expert_proj.weight)\n")
inc="".join("".join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile="a/"+p,tofile="b/"+p)) for p,a,b in [(model,mo,mn),(gather,old,new)])
(here/"incremental.diff").write_bytes(inc.encode())
def funcs(s,names):
    module=ast.parse(s)
    keep=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in names]
    for node in keep:node.decorator_list=[]
    return compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),"source","exec")
env={}
exec(funcs(get(norm),["_launch_config","_cluster_size_for_shape","_cluster_num_threads","_cluster_launch_config","norm_residual_launch_config","cluster_size_for_num_tokens"]),env)
env.update(_SMALL_M=192,_SMALL_M_THREAD_CHOICES=(256,512,1024,128),_THREAD_CHOICES=(1024,512,256,128),_CLUSTER_SIZES=(1,2,4,8,16),_VEC=8)
exec(funcs(new,["_boundary_config"]),env)
cases=[]
for sm in (9,10):
    env["_device_major"]=lambda:sm
    for m in (1,7,8,9,32,128,191,192,256,511,512):
        got=env["_boundary_config"](8192,m)
        golden=(1,*env["_launch_config"](8192,m)) if sm==9 else env["norm_residual_launch_config"](8192,m)
        assert got==golden
        cases.append(dict(sm=sm,rows=m,config=got))
receipt=dict(frozen_sha256=sha(patch.read_bytes()),base=base,increment_sha256=sha(inc.encode()),compiled_python=all(compile(s,"candidate","exec") for s in (mn,new)) is True,cases=cases,
    push_bytes_reserved=512*8192*2,per_rank_write_plus_read_avoided={m:4*m*8192 for m in (8,32,128,256)},
    parent_startup_log_truncated=judged["result"]["startup_log"]["truncated"])
(here/"dispatch.json").write_text(json.dumps(receipt,indent=2))
print(receipt["increment_sha256"],len(cases))
