"""Check the real packaged forward on the changed-input holdout outside timed pairs."""
from pathlib import Path
here=Path(__file__).resolve().parent
src=(here/"paired_fused.py").read_text()
pos=src.index("def case(m):")
hook='''from types import SimpleNamespace
module_name="sglang.kernels.ops.moe.muse_spark_grouped_routes"
packaged=fixture.module(module_name,here/"packaged_grouped.py")
owner_src=(here/"packaged_owner.py").read_text()
owner_method=next(n for cls in ast.parse(owner_src).body if isinstance(cls,ast.ClassDef)
                  for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_forward_marlin")
owner_method.decorator_list=[]
ns_pkg={"nn":torch.nn,"torch":torch}
from typing import Optional
ns_pkg["Optional"]=Optional
exec(compile(ast.Module(body=[owner_method],type_ignores=[]),"packaged_owner","exec"),ns_pkg)
report["package_sha256"]=hashlib.sha256((here/"candidate.diff").read_bytes()).hexdigest()
report["package_checks"]=[]
def verify_owner(env,x,ids,scores,expected,dist):
    norm_obj=object()
    lookup=torch.full((257,),-1,device="cuda",dtype=torch.int32)
    lookup[33:65]=torch.arange(32,device="cuda",dtype=torch.int32)
    owner=SimpleNamespace(use_humming=True,global_num_experts=256,_post_expert_norm=norm_obj,
        _post_expert_norm_weight_first=True,_post_expert_norm_eps=1e-8,_expert_map=lookup[1:],
        _expert_lookup=lookup,_humming_layer=layer)
    def failed(*a,**kw):raise RuntimeError("packaged fallback on qualified benchmark shape")
    owner._marlin_expert_outputs=failed
    gids=torch.where(ids>=0,ids+32,192)
    original=packaged.split if hasattr(packaged,"split") else None
    # Verify production address translation, scored epsilon, and both supported norm operand dtypes.
    for norm_dtype in (torch.bfloat16,torch.float32):
        owner._post_expert_norm_weight=norm.to(norm_dtype)
        try:out=ns_pkg["_forward_marlin"](owner,env.before(x),scores,gids,norm_obj)
        except Exception as exc:
            report["package_checks"].append(dict(m=x.shape[0],dist=dist,norm=str(norm_dtype),error=repr(exc)))
            save();return
        exact=torch.equal(env.after(out).view(torch.int16),expected.view(torch.int16))
        report["package_checks"].append(dict(m=x.shape[0],dist=dist,norm=str(norm_dtype),exact=exact))
        save()
        if not exact:raise RuntimeError("packaged full FFN parity")
'''
src=src[:pos]+hook+src[pos:]
look='        report["route_checks"].append(dict(rows=m,dist=dist,placed=len(placed),exact=compare(*outputs,dist)))'
assert src.count(look)==1
src=src.replace(look,look+'\n        if dist in ("all_remote","skewed"):verify_owner(env,x,ids,scores,outputs[0],dist)')
(here/"paired_hold.py").write_text(src)
launcher=(here/"launch_fused.py").read_text().replace('"paired_fused")','"paired_fused", "paired_hold")')
(here/"launch_hold.py").write_text(launcher)
watch=(here/"watchdog.py").read_text().replace('here/"launch_fused.py"','here/"launch_hold.py"').replace('"paired_fused"','"paired_hold"')
(here/"watch_hold.py").write_text(watch)
