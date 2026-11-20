"""Round-trip actual canonical/Humming words, selective writes, stage-scoped projection views."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from exclusive_pack import NativeFallback,active_experts,restore,install_layer
from test_group_prep import pack,scales

here=Path(__file__).resolve().parent
def expected(q,s):
    sq=s.transpose(0,2,1).reshape(s.shape[0],s.shape[2]//4,4,s.shape[1]).astype(np.uint32)
    a=np.bitwise_or.reduce(sq<<(np.arange(4,dtype=np.uint32)[None,None,:,None]*8),axis=2)
    return torch.from_numpy(q[:,:,::2]|(q[:,:,1::2]<<4)),torch.from_numpy(a.view(np.int32))


def check(seed):
    rng=np.random.default_rng(seed);e=5;h=128
    layer=torch.nn.Module();layer.humming_metas={}
    originals={}
    for name,n,wg in (("w13",2*h,True),("w2",h,False)):
        q=rng.integers(16,size=(e,n,h),dtype=np.uint8)
        sf=rng.integers(256,size=(e,n,h//16),dtype=np.uint8)
        native=pack(q,wg),scales(torch.from_numpy(sf))
        for suffix,v in zip(("weight","scale"),native):
            layer.register_parameter(name+"_"+suffix,torch.nn.Parameter(v,requires_grad=False))
        originals[name]=native,expected(q,sf)
        layer.humming_metas[name]=SimpleNamespace(weight_name=name+"_weight",weight_scale_name=name+"_scale",
                b_dtype="float4e2m1",bs_dtype="float8e4m3",weight_scale_group_size=16,has_zero_point=False,
                has_bias=False,should_apply_bs_on_c=False,use_fused_e8m0_scale=False,use_packed_k_layout=False,
                shape_n=n,shape_k=h,num_experts=e,mma_type="wgmma" if wg else "mma")
    old=sum(t.numel()*t.element_size() for t in layer.parameters())
    install_layer(layer)
    install_layer(layer)
    assert old==sum(t.numel()*t.element_size() for t in layer.parameters())
    for key,(_, (q,p)) in originals.items():
        assert torch.equal(q,getattr(layer,key+"_weight"))
        assert torch.equal(p,getattr(layer,key+"_scale"))
    # The checks retain originals externally only as reference data. The module owns two canonical banks.
    storages={x.untyped_storage().data_ptr() for x in layer.parameters()}
    assert len(storages)==2
    arena=NativeFallback(e*(2*h)*h*9//16,e,"cpu",workers=3)
    bytes_checked=0
    for active in ([],[0,4],[1,2,3],[0,1,2,3,4],[3]):
        ids=torch.full((34,8),-1,dtype=torch.int32)
        for j,v in enumerate(active):ids[(j*7+2)%34,(j*3)%8]=v
        ids[-1,-1]=e if not active else active[0]
        view=arena.view(layer,ids)
        assert arena.active[:len(active)].tolist()==active and int(arena.active[e])==len(active)
        for name in ("w13","w2"):
            arena.scratch.fill_(0x39)
            a=getattr(view,name+"_weight");b=getattr(view,name+"_scale")
            # A second light/heavy dispatch sees the same scratch without rematerializing.
            assert getattr(view,name+"_weight") is a
            for gold,got in zip(originals[name][0],(a,b)):
                np_gold=gold.view(torch.uint8).reshape(e,-1)
                np_obs=got.view(torch.uint8).reshape(e,-1)
                for expert in range(e):
                    if expert in active:
                        assert torch.equal(np_gold[expert],np_obs[expert]),(seed,name,expert)
                    else:
                        assert torch.all(np_obs[expert]==0x39),(seed,name,expert)
                    bytes_checked+=np_obs.shape[1]
    return dict(seed=seed,exact_bytes_checked=bytes_checked,resident_bytes=old,
                owned_storages=len(storages),shared_fallback_scratch_bytes=arena.scratch.numel())
if __name__=="__main__":
    assert os.environ.get("TRITON_INTERPRET")=="1"
    out=[check(seed) for seed in (31079,77161)]
    (here/"exclusive_checked.json").write_text(json.dumps(out,indent=2))
    print(out)
