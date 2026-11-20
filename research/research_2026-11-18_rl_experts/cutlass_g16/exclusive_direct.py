"""Normal indexed Humming fallback reading the exclusive FP4 layout, with selective restore for other kernels."""
import inspect
import json
from functools import lru_cache
import torch
from humming import ops
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod
from direct_loader import adapted_source


class DirectKernel(HummingKernel):
    vector=False
    scales_only=False
    def prepare(self):
        marker="#include <humming/kernel/humming.cuh>"
        assert self.code.count(marker)==1
        self.code=self.code.replace(marker,adapted_source("wgmma" in str(self.mma_type).lower(),self.vector,self.scales_only))
        super().prepare()


class DirectScalesKernel(DirectKernel):
    scales_only=True

class DirectVectorKernel(DirectKernel):
    vector=True

@lru_cache(maxsize=256)
def handles(context,meta_json,compute_json,tuning_json,vector,scales_only):
    c=json.loads(meta_json);c.pop("sublayer_name",None)
    c.update(json.loads(compute_json));c.update(json.loads(tuning_json))
    num_sms=c.pop("num_sms",0)
    cls=DirectScalesKernel if scales_only else DirectVectorKernel if vector else DirectKernel
    obj=cls(**c)
    return torch.tensor([obj.kernel_id,num_sms],dtype=torch.int64,device="cpu")


class CanonicalMethod:
    calls={}
    vector=False
    @classmethod
    def forward_layer(cls,view,**kw):
        c=kw["tuning_config"];compute=kw["compute_config"];stage=kw["sublayer_name"]
        layer=view._layer
        scales_only=hasattr(layer,"_shared_scale_words")
        info=(layer._shared_scale_words if scales_only else layer._exclusive_cutlass)[stage]
        m=layer.humming_metas[stage]
        allowed=(kw["inputs"].dtype==torch.bfloat16 and c.get("use_stream_k") is False and
                 c.get("use_f16_accum") is False and not c.get("use_warp_spec") and not c.get("use_tma")
                 and not c.get("use_mbarrier") and not any(v for k,v in c.items() if k.startswith("use_tma_"))
                 and c.get("multi_cast_size_a",1)==c.get("multi_cast_size_b",1)==1
                 and compute.get("gemm_type")=="indexed" and compute.get("use_f16_accum") is False
                 and kw["top_k"] in (1,8) and kw["valid_shape_m"]>0 and not m.has_input_scale)
        vector=cls.vector and c["block_shape"][1]%64==0 and c["block_shape"][2]%64==0
        allowed=allowed and adapted_source(info.transpose,vector,scales_only) is not None
        key=stage+("_scales" if allowed and scales_only else "_vector" if allowed and vector else "_direct" if allowed else "_restored")
        cls.calls[key]=cls.calls.get(key,0)+1
        if not allowed:
            return HummingMethod.forward_layer(view,**kw)
        code=handles(HummingKernel.current_context(),m.to_str(),json.dumps(compute,sort_keys=True),
                     json.dumps(c,sort_keys=True),vector,scales_only)
        # Keep Humming's Python/C++ launch ABI (native shape and dtype). The bytes are exclusively canonical.
        w=getattr(layer,info.weight_name).view(info.dtype_w).reshape(info.shape_w)
        s=getattr(layer,info.scale_name).view(info.dtype_s).reshape(info.shape_s)
        return ops.launch_kernel(configs=code,inputs=kw["inputs"],outputs=kw["outputs"],weight=w,weight_scale=s,
                  weight_scale_2=getattr(layer,m.weight_scale_2_name,None),sorted_ids=kw["sorted_ids"],
                  expert_ids=kw["expert_ids"],num_tokens_padded=kw["num_tokens_padded"],
                  locks=getattr(layer,"locks",None),top_k=kw["top_k"],valid_shape_m=kw["valid_shape_m"])


def bind_frozen(default,split):
    """Bind only the weight-loader callback in local copies of exact existing helpers for the A/B test."""
    def clone(fn,scope,change):
        src=inspect.getsource(fn)
        needle="from humming.layer import HummingMethod"
        if change:
            assert src.count(needle)==1
            src=src.replace(needle,"from exclusive_direct import CanonicalMethod as HummingMethod")
        exec(compile(src,fn.__code__.co_filename+":canonical", "exec"),scope)
        return scope[fn.__name__]
    ds=default.__globals__.copy()
    wrapped_default=clone(default,ds,True)
    ss=split.__dict__.copy()
    ss["_original_split_expert_outputs"]=clone(split._original_split_expert_outputs,ss,True)
    ss["_split_large_prefill_expert_outputs"]=clone(split._split_large_prefill_expert_outputs,ss,True)
    wrapped_split=clone(split.split_expert_outputs,ss,False)
    def run(view,x,ids,torch,*,defer_downscale,global_experts):
        return wrapped_split(wrapped_default,view,x,ids,torch,global_experts=global_experts,defer_downscale=defer_downscale)
    return run
