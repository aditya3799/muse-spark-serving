"""Verify startup storage, per-stage scale traffic and complete native scale fallback on CPU."""
import json
import os
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np

from test_group_prep import pack,scales
from test_exclusive import expected
from exclusive_pack import NativeFallback
from shared_scales import install_scales,words_from_native

def check(seed):
    rng=np.random.default_rng(seed);e=4;h=128
    layer=torch.nn.Module();layer.humming_metas={}
    orig={};ptr={}
    for name,n,wg in (("w13",2*h,True),("w2",h,False)):
        q=rng.integers(16,size=(e,n,h),dtype=np.uint8)
        sf=rng.integers(256,size=(e,n,h//16),dtype=np.uint8)
        w=pack(q,wg);s=scales(torch.from_numpy(sf))
        ptr[name]=w.data_ptr()
        layer.register_parameter(name+"_w",torch.nn.Parameter(w,requires_grad=False))
        layer.register_parameter(name+"_s",torch.nn.Parameter(s,requires_grad=False))
        layer.humming_metas[name]=SimpleNamespace(weight_name=name+"_w",weight_scale_name=name+"_s",b_dtype="float4e2m1",
            bs_dtype="float8e4m3",weight_scale_group_size=16,has_bias=False,has_zero_point=False,
            should_apply_bs_on_c=False,use_fused_e8m0_scale=False,use_packed_k_layout=False,shape_n=n,shape_k=h,
            num_experts=e,mma_type="wgmma" if wg else "mma")
        orig[name]=(s,expected(q,sf)[1])
    before=sum(t.untyped_storage().nbytes() for t in layer.parameters())
    install_scales(layer);install_scales(layer)
    after=sum(t.untyped_storage().nbytes() for t in layer.parameters())
    assert before==after
    arena=NativeFallback(e*2*h*h//16,e,"cpu",workers=3)
    checked=0
    for selected in ([],[0,3],[1,2],list(range(e)),[1]):
        ids=torch.full((39,8),-1,dtype=torch.int32)
        for j,expert in enumerate(selected):ids[7+j,2]=expert
        view=arena.view(layer,ids,lazy=True)
        for name in ("w13","w2"):
            assert getattr(view,name+"_w").data_ptr()==ptr[name]
            if name=="w13":assert view._pending is ids
            arena.scratch.fill_(0x55)
            actual=getattr(view,name+"_s")
            assert torch.equal(getattr(layer,name+"_s"),orig[name][1])
            for expert in range(e):
                z=actual[expert].view(torch.uint8)
                if expert in selected:assert torch.equal(z,orig[name][0][expert].view(torch.uint8))
                else:assert torch.all(z==0x55)
                checked+=z.numel()
    return dict(seed=seed,exact_scale_bytes=checked,original_weight_storage_pointers_preserved=True,
                resident_bytes_before=before,resident_bytes_after=after,scale_fallback_workspace_bytes=arena.scratch.numel())
if __name__=="__main__":
    assert os.environ.get("TRITON_INTERPRET")=="1"
    out=[check(s) for s in (71273,89459)]
    Path(__file__).with_name("shared_scales_checked.json").write_text(json.dumps(out,indent=2))
    print(out)
