from pathlib import Path
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe')
s=(p/'humming_probe.py').read_text(encoding='utf-8-sig')
s=s.replace("torch.zeros(1024,device='cuda'", "torch.zeros(max(2048,torch.cuda.get_device_properties(0).multi_processor_count*8),device='cuda'")
s=s.replace("    valid=local_ids.numel()", "    for intervals in configs.values():\n        for lo,hi,c in intervals:\n            c['use_stream_k']=False\n            c['use_f16_accum']=False\n    valid=local_ids.numel()")
s=s.replace("    block=selected['w13']['block_shape'][0]", "    assert all(c['use_stream_k'] is False and c['use_f16_accum'] is False for c in selected.values())\n    block=selected['w13']['block_shape'][0]")
s=s.replace("    cache1=torch.empty((m*topk,2*i),device=x.device,dtype=x.dtype)", "    cache13=torch.empty((m*topk*max(2*i,k),),device=x.device,dtype=x.dtype)\n    cache1=cache13[:m*topk*2*i].view(m*topk,2*i)")
s=s.replace("    cache3=torch.empty((m*topk,k),device=x.device,dtype=x.dtype)", "    cache3=cache13[:m*topk*k].view(m*topk,k)")
(p/'humming_probe.py').write_text(s)
header='''# okc: problem=muse_moe_w4a16
# Uses Apache-2.0 licensed Humming and SGLang public APIs.
"""NVFP4 W4A16 expert GEMMs with FP32 accumulation and no Stream-K.

One canonical packed weight representation per expert. Global scales are
applied after each BF16 GEMM output, matching the stock Marlin boundary.
"""
from typing import TypedDict
from types import SimpleNamespace
import torch
try:
    from humming.layer import HummingMethod
except ImportError as exc:
    raise RuntimeError("This candidate requires the judge-pinned humming-kernels 0.1.12 dependency") from exc
try:
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
except ImportError:
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum

class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor

class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str

class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config

'''
functions=s[s.index('def prepare_humming'):s.index('def main():')]
footer='''def prepare(weights):
    # The task supplies global weights before a per-call shard config. Store each
    # expert once, so any contiguous shard can be selected without caching inputs.
    raw=SimpleNamespace()
    for prefix in ('w13','w2'):
        setattr(raw,prefix+'_weight',weights[prefix])
        setattr(raw,prefix+'_weight_scale',weights[prefix+'_sf'])
        setattr(raw,prefix+'_weight_scale_2',weights[prefix+'_gs'])
    layer=prepare_humming(raw,torch)
    return {'layer':layer,'norm_w':(weights['post_norm_w'].float()+1.0).to(torch.bfloat16)}


def kernel_fn(data,state):
    cfg=data['config']; x=data['hidden_states'].contiguous()
    ids=data['topk_ids'].to(torch.int32).contiguous()
    active=(ids>=cfg['local_start']) & (ids<cfg['local_start']+cfg['local_count'])
    global_ids=torch.where(active,ids,-1)
    outputs,valid_ids=indexed(state['layer'],x,global_ids,torch)
    return fused_post_expert_norm_weighted_sum(outputs,data['topk_weights'].to(torch.float32).contiguous(),
        norm_weight=state['norm_w'],eps=1e-8,weight_first=True,slot_expert_ids=valid_ids)
'''
(p/'submission_humming_fp32_v1.py').write_text(header+functions+footer)
print('adapter written',len(header+functions+footer))
