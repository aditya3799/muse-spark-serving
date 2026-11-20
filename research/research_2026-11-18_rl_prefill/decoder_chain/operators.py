"""Shared single-rank Muse decoder operators for the integrated comparison."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch.nn.functional as F

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
EXPERTS = ROOT/"research_2026-11-18_rl_experts"
SLAB = ROOT/"research_2026-11-18_rl/expert_slab"
sys.path.insert(0,str(SLAB))
import gpu_probe as hg
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
reader=load_module("chain_qkv_reader",H.parent/"decode_qkv/bench.py")
QKDIR=H.parent/"decode_qkv"
oldq=reader.qk(QKDIR/"parent_muse_spark_qk_norm_rope_store.py")
newq=reader.qk(QKDIR/"muse_spark_qk_norm_rope_store.py")
prepare=reader.extract(QKDIR/"parent_flashattention_backend.py","FlashAttentionBackend",["prepare_paged_mha_query"],dict(torch=torch)).prepare_paged_mha_query
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.layernorm.muse_spark_norm_gate_jit import rms_norm_gate
from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import fused_norm_residual
from sglang.kernels.ops.moe.muse_spark_routing_jit import fused_muse_spark_routing_jit
finish=hg.fixture.module("chain_scored_finalize",hg.ops/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
split=sys.modules["sglang.kernels.ops.moe.muse_spark_humming_split"]
group=load_module("chain_grouped",EXPERTS/"grouped_route/grouped_fused.py")
b16=torch.bfloat16
f8=torch.float8_e4m3fn

def bank():
    raw=torch.nn.Module()
    for prefix,n in (("w13",8192),("w2",4096)):
        for key,t in (("weight",torch.randint(0,256,(32,n,2048),device="cuda",dtype=torch.uint8)),
                      ("weight_scale",torch.randint(40,68,(32,n,256),device="cuda",dtype=torch.uint8).view(f8)),
                      ("weight_scale_2",torch.linspace(.125,.5,32,device="cuda").bfloat16())):
            setattr(raw,prefix+"_"+key,torch.nn.Parameter(t,requires_grad=False))
    with hg.cached_threshold_for_preparation(H):layer=hg.ns["_prepare_humming_experts"](raw,torch)
    return layer

def bfmat(n,k):
    return torch.empty((n,k),device="cuda",dtype=b16).normal_(std=1/(k**.5))
class Block:
    def __init__(self,m,pages,j):
        self.j=j;self.full=j==1
        self.wq,self.wo,self.pool,self.unpool=bfmat(2304,8192),bfmat(8192,1024),bfmat(4096,8192),bfmat(8192,4096)
        self.route_w=torch.empty(256,8192,device="cuda",dtype=torch.float32).normal_(std=.012)
        hi=self.route_w.bfloat16();lo=(self.route_w-hi.float()).bfloat16()
        self.route_split=torch.cat((hi,lo)).t().contiguous()
        self.bias=torch.zeros(256,device="cuda")
        self.norm=torch.linspace(.85,1.15,4096,device="cuda").bfloat16()
        self.final_norm=torch.linspace(.7,1.2,4096,device="cuda").bfloat16()
        self.pre=torch.linspace(.9,1.1,8192,device="cuda").bfloat16()
        gate=torch.randn(8192,device="cuda")*.15/.3
        self.beta=gate.sigmoid()
        self.alpha=((-gate).sigmoid()*(1.+self.beta)).clamp(min=1e-3).sqrt()
        self.pooled=torch.empty((m,4096),device="cuda",dtype=b16)
        self.expanded=torch.empty((m,8192),device="cuda",dtype=b16)
        self.k8=bfmat(pages,128).mul_(4).to(f8);self.v8=bfmat(pages,128).mul_(4).to(f8)
        self.k=self.k8.to(b16).view(-1,1,2,64);self.v=self.v8.to(b16).view(-1,1,2,64)
    def entry(self,hidden,pos,rotary,loc,table,new):
        packet=F.linear(hidden,self.wq)
        op=NS(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1.e-5)
        pa,ro=(None,None) if self.full else (pos,rotary)
        if new:
            q,_=newq(op,packet,pa,ro,(self.k8,self.v8,loc,table))
        else:
            q,k=oldq(op,packet,pa,ro)
            store_cache(k.to(f8).view(torch.uint8),packet[:,1152:1280].to(f8).view(torch.uint8),
                        self.k8.view(torch.uint8),self.v8.view(torch.uint8),loc,row_bytes=128,v_row_bytes=128,size_limit=self.k8.shape[0])
        backend=NS(kv_cache_dtype_str="fp8_e4m3",kv_cache_is_mxfp8=False,kv_cache_dtype=f8,fa_impl_ver=3)
        q=prepare(backend,q,None,None,NS(head_dim=64,k_scale=None,v_scale=None),logical_batch_size=len(pos)//8,kv_head_num=2,is_prefill=True)[0]
        # Adapter for the native SM86 attention core. H200 would directly use
        # the three E4M3 tensors. Both timed arms execute this identical bridge.
        self.k.view(-1,128).index_copy_(0,loc,self.k8.view(torch.uint8)[loc].view(f8).to(b16))
        self.v.view(-1,128).index_copy_(0,loc,self.v8.view(torch.uint8)[loc].view(f8).to(b16))
        return q.to(b16).view(-1,16,64),packet[:,-1024:]
    def finish(self,attn,gate,stream,layer,side,stage,grouped=False):
        o=F.linear(rms_norm_gate(attn.reshape(-1,1024),gate,None,head_dim=64,eps=1e-5,has_norm=True),self.wo)
        m=o.shape[0];use_split=m>8
        ff,stream,route=fused_norm_residual(o,stream,1e-8,1e-5,pre_weight=self.pre,alpha=self.alpha,beta=self.beta,want_router=not use_split)
        main=torch.cuda.current_stream();side.wait_stream(main)
        if stage:self.pooled.copy_(torch.mm(ff,self.pool.t(),out_dtype=torch.float32).bfloat16())
        self.pooled[:,:512].copy_(torch.mm(ff,self.pool[:512].t(),out_dtype=torch.float32).bfloat16())
        inner=F.rms_norm(self.pooled,(4096,),self.norm,1e-5)
        with torch.cuda.stream(side):
            if use_split:
                hi,lo=torch.mm(ff,self.route_split,out_dtype=torch.float32).chunk(2,dim=-1)
                logits=hi+lo
            else:logits=F.linear(route,self.route_w)
            scores,ids,_=fused_muse_spark_routing_jit(logits,self.bias,8,1.,.17677669529663687,router_score_division_eps=1e-15,
                packed_topk_ids=torch.empty((m,8),device="cuda",dtype=torch.int32))
            local=torch.where(ids<32,ids,-1)
        main.wait_stream(side)
        if grouped:
            part=group.compute(layer,inner,local,split,"split_fused",scores,self.final_norm)
        else:
            down,slots=hg.ns["_humming_expert_outputs"](layer,inner,local,torch,global_experts=256,defer_downscale=True)
            part=finish(down,scores,norm_weight=self.final_norm,eps=1e-8,weight_first=True,slot_expert_ids=slots,expert_output_scale=layer.w2_post_scale)
        if stage:self.expanded.copy_(torch.mm(part,self.unpool.t(),out_dtype=torch.float32).bfloat16())
        self.expanded[:,:1024].copy_(torch.mm(part,self.unpool[:1024].t(),out_dtype=torch.float32).bfloat16())
        nex,stream,_=fused_norm_residual(self.expanded,stream,1e-8,1e-5,post_norm_is_input_scale=True,post_weight=self.pre,
                    pre_weight=self.pre,alpha=self.alpha,beta=self.beta)
        return nex,stream,ids
