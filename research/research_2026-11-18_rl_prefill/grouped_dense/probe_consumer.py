"""Execute the actual prototype runner on CPU with only native CUDA methods replaced."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S
import numpy as np
import torch
import consumer
from qualify import phase
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/"research_2026-11-18_rl/expert_slab"))
from test_slab import pack,scales
from slab import make_table, materialize

def norm_scale(rows,ids,gs):
    valid=ids.flatten()>=0
    rows[valid]=rows[valid]*gs[ids.flatten()[valid].long(),None]

def silu(gu,out,expert_ids,expert_step):
    gate,value=gu.chunk(2,dim=1)
    out.copy_((torch.nn.functional.silu(gate.float())*value.float()).bfloat16())

fn=next(n for n in ast.parse((H/"consumer.py").read_text()).body if getattr(n,"name",None)=="run")
dropped=[]
def keep(node):
    if isinstance(node,(ast.Import,ast.ImportFrom)):
        dropped.append(ast.unparse(node))
        return False
    if isinstance(node,ast.Assert) and "get_device_capability" in ast.unparse(node.test):
        dropped.append(ast.unparse(node))
        return False
    return True
fn.body=[n for n in fn.body if keep(n)]
scope=dict(consumer.__dict__)
scope.update(deep_gemm=S(get_mk_alignment_for_contiguous_layout=lambda:16),
             silu_and_mul=silu,scale_active_rows=norm_scale,materialize=materialize,
             native_consumer=lambda a,b,d,ends: phase(a,b,d,S(group_ends=ends),16))
exec(compile(ast.Module([fn],[]),str(H/"consumer.py"),"exec"),scope)
rows=[]
for seed,trans,chunk in [(3453,False,3),(7549,True,4)]:
    rng=np.random.default_rng(seed)
    gen=torch.Generator().manual_seed(seed)
    h=64;e=5;k=4;t=9;slab=3
    layer=S(w13_post_scale=torch.linspace(.35,1.17,e).bfloat16(),
            w2_post_scale=torch.linspace(.65,1.33,e).bfloat16(),humming_metas={})
    math_weights={}
    lut=make_table("cpu")
    for stage,n in (("w13",2*h),("w2",h)):
        code=rng.integers(0,16,(e,n,h),dtype=np.uint8)
        sf=torch.from_numpy(rng.integers(25,70,(e,n,h//16),dtype=np.uint8))
        pw=pack(code,trans); ps=scales(sf)
        setattr(layer,stage+"_packed",pw);setattr(layer,stage+"_sf",ps)
        layer.humming_metas[stage]=S(should_apply_bs_on_c=False,use_fused_e8m0_scale=False,weight_scale_group_size=16,
                                      weight_name=stage+"_packed",weight_scale_name=stage+"_sf",mma_type="wgmma" if trans else "mma")
        math_weights[stage]=lut[sf.long().repeat_interleave(16,dim=2),torch.from_numpy(code).long()]
    workspace=consumer.Workspace.allocate("cpu",h,slab,chunk,k,16,make_table)
    base_ptrs=tuple(a.data_ptr() for a in vars(workspace).values())
    for defer in [False,True]:
        x=torch.randn((t,h),generator=gen).bfloat16()
        ids=torch.randint(-2,e+2,(t,k),generator=gen,dtype=torch.int32)
        ids[0]=torch.tensor([0,0,4,-1])
        y=torch.full((t,k,h),-177.,dtype=x.dtype)
        ref=y.clone()
        for g in range(e):
            ix=torch.where(ids.flatten()==g)[0]
            if not len(ix):continue
            gu=(x[ix//k].float()@math_weights["w13"][g].float().T).bfloat16().mul_(4).mul_(layer.w13_post_scale[g])
            a,b=gu.chunk(2,dim=1)
            activation=(torch.nn.functional.silu(a.float())*b.float()).bfloat16()
            d=(activation.float()@math_weights["w2"][g].float().T).bfloat16().mul_(4)
            if not defer:d.mul_(layer.w2_post_scale[g])
            ref.view(-1,h)[ix]=d
        actual,returned_ids=scope["run"](layer,x,ids,y,workspace,defer_downscale=defer,token_chunk=chunk,slab_size=slab)
        assert returned_ids is ids and actual is y and torch.equal(actual.view(torch.int16),ref.view(torch.int16))
        assert base_ptrs==tuple(a.data_ptr() for a in vars(workspace).values())
        rows.append(dict(seed=seed,defer_downscale=defer,mini_transpose=trans,output_bf16_elements=ref.numel(),
                         remote_slots=int(((ids<0)|(ids>=e)).sum()),exact=True))
out=dict(replaced_imports_and_device_check=dropped,substitutions=["native grouped BF16 GEMM -> grouped CPU FP32 accumulation and BF16 store",
         "CUDA post-scale and fused activation -> CPU BF16 scale and FP32 SiLU product"],
         actual="unchanged runner statements, reused workspace, true Humming forward packer and inverse converter/Triton CPU gather/scatter, original slot indices",
         results=rows)
(H/"runner_cpu.json").write_text(json.dumps(out,indent=2))
print("Runner",rows)

