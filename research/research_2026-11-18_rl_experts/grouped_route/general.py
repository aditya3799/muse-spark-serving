"""Build the geometry-based research composite with no dispatch by benchmark row count."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
here=Path(__file__).resolve().parent
root=here.parents[1];repo=root/"sglang";base="297e54fd105e974cdf401e5b4d20253a22c16f90"
prefix="python/sglang/"
gn=prefix+"kernels/ops/moe/muse_spark_grouped_routes.py"
on=prefix+"srt/layers/muse_spark/muse_spark_fp4_moe.py"
an=prefix+"kernels/ops/moe/route_activation.py"
sn=prefix+"kernels/ops/moe/muse_spark_humming_split.py"
pn=prefix+"kernels/ops/moe/muse_spark_route_policy.py"
policy=(here/"route_regime.py").read_text()
sha=lambda a:hashlib.sha256(a).hexdigest()
with tempfile.TemporaryDirectory(dir=here) as td:
    tmp=Path(td);objects=tmp/"objects";objects.mkdir()
    env=dict(os.environ,GIT_INDEX_FILE=str(tmp/"index"),GIT_OBJECT_DIRECTORY=str(objects),GIT_ALTERNATE_OBJECT_DIRECTORIES=str(repo/".git/objects"))
    def git(*args,data=None):
        return subprocess.run(["git","-C",str(repo),*args],input=data,env=env,capture_output=True,check=True).stdout
    def put(name,src):
        ast.parse(src);obj=git("hash-object","-w","--stdin",data=src.encode()).decode().strip()
        git("update-index","--add","--cacheinfo","100644",obj,name)
    def apply(p):git("apply","--cached","--whitespace=error","-",data=p)
    git("read-tree",base);apply((root/"release_2026-11-18/candidate.diff").read_bytes());parent=git("write-tree").decode().strip()
    git("read-tree",base);apply((here/"joint_16k.diff").read_bytes())
    group=git("show",":"+gn).decode()
    group=group.replace("def compute(layer,x,ids,split,variant,scores,norm):","def compute(layer,x,ids,split,variant,scores,norm,global_experts):")
    group=group.replace("split._choose_large_prefill_configs(layer,slots,256)","split._choose_large_prefill_configs(layer,slots,global_experts)")
    group=group.replace("x.shape[0] not in (128,768) or x.shape[1]!=4096 or owner.global_num_experts!=256","x.shape[1]!=4096 or owner.global_num_experts<32 or owner.global_num_experts%32")
    group=group.replace("plan=split._choose_large_prefill_configs(layer,topk_ids.numel(),256)\n    if plan is None:","from sglang.kernels.ops.moe.muse_spark_route_policy import route_regime\n    plan=split._choose_large_prefill_configs(layer,topk_ids.numel(),owner.global_num_experts)\n    if route_regime(topk_ids.numel(),32,owner.global_num_experts,plan) != \"grouped\":")
    group=group.replace('scores.contiguous(),norm.contiguous())','scores.contiguous(),norm.contiguous(),owner.global_num_experts)')
    owner=git("show",":"+on).decode().replace("if defer_downscale and hidden_states.shape[0] in (128, 768):","if defer_downscale:")
    act=git("show",":"+an).decode().replace("def scaled_prefill_activation(x, out, ids, scales):","def scaled_prefill_activation(x, out, ids, scales, *, dense_routes=False):").replace("ids.ndim == 2 and ids.shape[0] == 16384 and ids.shape[1] == 8","dense_routes and ids.ndim == 2 and ids.shape[1] == 8")
    split=git("show",":"+sn).decode()
    split=split.replace("    scaled_prefill_activation(cache1, cache2, local_ids, layer.w13_post_scale)",'''    from sglang.kernels.ops.moe.muse_spark_route_policy import route_regime
    dense_routes = route_regime(valid, experts, global_experts, plan) == "indexed_large"
    scaled_prefill_activation(cache1, cache2, local_ids, layer.w13_post_scale, dense_routes=dense_routes)''')
    assert split.count("dense_routes=dense_routes")==2
    assert "owner.global_num_experts,plan)" in group and "x.shape[0] not in (128,768)" not in group
    assert "ids.shape[0] == 16384" not in act and "hidden_states.shape[0] in (128, 768)" not in owner
    helper_name=prefix+"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
    helper=git("show",":"+helper_name).decode()
    helper=helper.replace('def _humming_expert_outputs_default(layer, x, local_ids, torch, retain=False, *, defer_downscale=False):','def _humming_expert_outputs_default(layer, x, local_ids, torch, retain=False, *, defer_downscale=False, global_experts=None):')
    helper=helper.replace('    scaled_prefill_activation(cache1, cache2, local_ids, layer.w13_post_scale)','    from sglang.kernels.ops.moe.muse_spark_route_policy import dense_route_tiles\n    dense_routes = dense_route_tiles(valid, global_experts, selected)\n    scaled_prefill_activation(cache1, cache2, local_ids, layer.w13_post_scale, dense_routes=dense_routes)')
    helper=helper.replace('    return split_expert_outputs(\n        _humming_expert_outputs_default,layer,x,local_ids,torch,global_experts,','    from functools import partial\n    default = partial(_humming_expert_outputs_default, global_experts=global_experts)\n    return split_expert_outputs(\n        default,layer,x,local_ids,torch,global_experts,')
    assert 'default = partial(' in helper and 'dense_routes=dense_routes' in helper
    for name,src in ((gn,group),(on,owner),(an,act),(sn,split),(pn,policy),(helper_name,helper)):
        put(name,src);(here/("general_"+Path(name).name)).write_text(src)
    target=git("write-tree").decode().strip()
    for name,start in (("general_joint.diff",base),("general_joint_delta.diff",parent)):
        patch=git("diff","--binary","--no-renames",start,target)
        git("read-tree",start);apply(patch);assert git("write-tree").decode().strip()==target
        (here/name).write_bytes(patch)
    (here/"general_joint.json").write_text(json.dumps(dict(base=base,parent=parent,target=target,sha256=sha((here/"general_joint.diff").read_bytes()),
        eligibility="Indexed large if global-average routes per expert exceed the heavy GEMM tile. Otherwise grouped only when its extra worst-case light arena is <=50% of incumbent activation scratch and all geometry/format guards admit. All other cases are indexed.",
        status="Research composite. Off-grid integration and paired timings pending."),indent=2))
    print(sha((here/"general_joint.diff").read_bytes()))
