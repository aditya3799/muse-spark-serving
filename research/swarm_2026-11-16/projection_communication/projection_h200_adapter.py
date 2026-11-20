# okc: problem=muse_bf16_proj
"""Full-output single-GPU adapter for FP32-output projection column shards.

All required output columns are computed on every invocation. The serial
assembly validates the distributed candidate's arithmetic, not its network
behavior or end-to-end serving speedup. No activation-derived state is cached.
"""
from typing import TypedDict
import torch

class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor

class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str

class Inputs(TypedDict):
    x: torch.Tensor
    config: Config

def prepare(weights):
    state = dict(weights)
    high = weights['router'].to(torch.bfloat16)
    low = (weights['router'] - high.float()).to(torch.bfloat16)
    state['router_high'] = high
    state['router_low'] = low
    return state

def kernel_fn(data, state):
    x = data['x']
    config = data['config']
    if config['layer'] == 'router':
        high = torch.mm(x, state['router_high'].t(), out_dtype=torch.float32)
        low = torch.mm(x, state['router_low'].t(), out_dtype=torch.float32)
        return high + low
    weight = state[config['layer']][:config['n']]
    n, k = weight.shape
    if n >= 4096 and k >= 4096 and n % 8 == 0:
        width = n // 8
        parts = []
        for start in range(0, n, width):
            value = torch.mm(
                x, weight[start:start + width].t(), out_dtype=torch.float32
            )
            parts.append(value.to(x.dtype))
        return torch.cat(parts, dim=1)
    return torch.mm(x, weight.t(), out_dtype=torch.float32).to(x.dtype)
