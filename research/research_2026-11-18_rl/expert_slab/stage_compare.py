"""Trace the original two Humming phases and compare BFloat16 matrix boundaries."""
import torch
from humming.layer import HummingMethod
from ordered_dense import ordered

def difference(a,b):
    av=a.float();bv=b.float()
    return dict(unequal_bits=int((a.view(torch.int16)!=b.view(torch.int16)).sum()),values=a.numel(),
                max_abs=float((av-bv).abs().max()),relative_l2=float(torch.linalg.vector_norm(av-bv)/torch.linalg.vector_norm(av).clamp_min(1e-20)))

def examine(parent,decode,rows):
    descriptor=HummingMethod.__dict__.get("forward_layer")
    bound=HummingMethod.forward_layer
    calls={}
    def observe(layer,*args,**kwargs):
        stage=kwargs["sublayer_name"]
        ret=bound(layer,*args,**kwargs)
        if int(kwargs["num_tokens_padded"].item())>0 and bool((kwargs["expert_ids"]==0).any()):
            src=kwargs["inputs"] if stage=="w13" else kwargs["inputs"][::8]
            assert src.shape[0]==rows
            cfg=kwargs["tuning_config"]
            calls[stage]=dict(inp=src.contiguous().clone(),ref=kwargs["outputs"][::8].contiguous().clone(),
                              bk=cfg["block_shape"][2],wk=cfg["warp_shape"][2],block_shape=cfg["block_shape"],warp_shape=cfg["warp_shape"])
        return ret
    try:
        HummingMethod.forward_layer=staticmethod(observe)
        final=parent().clone()
    finally:
        if descriptor is None:delattr(HummingMethod,"forward_layer")
        else:HummingMethod.forward_layer=descriptor
    assert set(calls)=={"w13","w2"} and torch.equal(final,calls["w2"]["ref"])
    out={}
    for stage,call in calls.items():
        # Contiguous separated output features span the weight matrix while bounding diagnostic cost.
        n=call["ref"].shape[1]
        indices=torch.linspace(0,n-1,256,device="cuda").long()
        w=decode(stage).index_select(0,indices)
        ref=call["ref"].index_select(1,indices)
        x=call["inp"];bk,wk=call["bk"],call["wk"]
        cu={}
        initial=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            for setting in (True,False):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=setting
                cu[str(setting)]=difference(ref,torch.mm(x,w.T).mul_(4))
        finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=initial
        seen={}
        for dk,dw in sorted({(bk,wk),(64,64),(64,32),(128,32),(128,64)}):
            seen[f"{dk}/{dw}"]=difference(ref,ordered(x,w,dk,dw))
        out[stage]=dict(block_shape=call["block_shape"],warp_shape=call["warp_shape"],dense_mm=cu,ordered=seen)
    return out
