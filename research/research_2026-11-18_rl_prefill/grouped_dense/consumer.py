"""Experimental SM90 BF16 grouped consumer for the scored Humming expert weights.

Call only after qualifying the Humming output-bit boundary for the chosen native
GEMM. Every row plan has a fixed device allocation and variable device endpoints.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

@dataclass
class Plan:
    source_slot: torch.Tensor
    dense_row: torch.Tensor
    row_expert: torch.Tensor
    group_ends: torch.Tensor

def round_up(v, a):
    return (v + a - 1) // a * a

def routes(ids, first, groups, alignment):
    r = ids.numel()
    capacity = round_up(r + groups * (alignment - 1), alignment)
    ix = ids.flatten().to(torch.int32) - first
    keys, order = torch.sort(torch.where((ix >= 0) & (ix < groups), ix, groups), stable=True)
    cuts = torch.searchsorted(keys, torch.arange(groups + 1, device=ids.device, dtype=torch.int32))
    counts = cuts.diff()
    padded = round_up(counts, alignment)
    starts = torch.cumsum(padded, 0) - padded
    ends = starts + counts
    ranks = torch.arange(r, device=ids.device)
    e = keys.long().clamp_max(groups - 1)
    dest = torch.where(keys < groups, starts[e] + ranks - cuts[e], -1).int()
    rows = torch.arange(capacity, device=ids.device)
    g = torch.searchsorted(ends, rows, right=True)
    idx = g.clamp_max(groups - 1)
    live = (g < groups) & (rows >= starts[idx]) & (rows < ends[idx])
    return Plan(order.int(), dest, torch.where(live, idx + first, -1).int(), ends.int())

@triton.jit
def _move(X, Y, Slots, Rows, WIDTH: tl.constexpr, TOPK: tl.constexpr,
          PACK: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0)
    row = tl.load(Rows + i)
    if row < 0:
        return
    slot = tl.load(Slots + i)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    source = slot // TOPK if PACK else row
    target = row if PACK else slot
    val = tl.load(X + source.to(tl.int64) * WIDTH + col, mask=col < WIDTH, other=0.)
    tl.store(Y + target.to(tl.int64) * WIDTH + col, val, mask=col < WIDTH)

def move(x, y, p, k, pack):
    width = x.shape[-1]
    _move[(p.source_slot.numel(), triton.cdiv(width, 128))](
        x, y, p.source_slot, p.dense_row, width, k, pack, 128)

def memory_budget(h, g, token_chunk, k, alignment):
    cap = round_up(token_chunk*k + g*(alignment - 1), alignment)
    weights = g * (2*h*h+h*h) * 2
    rows = cap * (h+2*h+h) * 2
    return dict(padded_rows=cap, bf16_weight_bytes=weights, bf16_row_bytes=rows,
                bf16_total_bytes=weights+rows)

@dataclass
class Workspace:
    w13: torch.Tensor
    w2: torch.Tensor
    io: torch.Tensor
    up: torch.Tensor
    act: torch.Tensor
    lut: torch.Tensor

    @classmethod
    def allocate(cls, device, h, slab_size, token_chunk, k, alignment, make_table):
        # The runner owns this single workspace across sequential layers/graphs.
        # Callers must serialize its use by stream, as with the model's KV graph buffers.
        mem = memory_budget(h, slab_size, token_chunk, k, alignment)
        if mem["bf16_total_bytes"] + (16 << 20) > 1 << 30:
            raise ValueError("Slab geometry exceeds the 1-GiB scratch budget")
        opt = dict(device=device, dtype=torch.bfloat16)
        cap = mem["padded_rows"]
        return cls(torch.empty((slab_size,2*h,h),**opt), torch.empty((slab_size,h,h),**opt),
                   torch.empty((cap,h),**opt), torch.empty((cap,2*h),**opt),
                   torch.empty((cap,h),**opt), make_table(device))

def native_consumer(a, b, d, ends):
    import deep_gemm
    # The SM90 scheduler reads the small device vector and skips absent groups.
    # Group ends count actual rows from alignment-padded starts, not padded ends.
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, ends, use_psum_layout=True)

def run(layer, x, ids, out, workspace, *, defer_downscale, token_chunk=512, slab_size=8):
    """out is the caller-owned [tokens, topk, width] result. Remote slots stay untouched."""
    import deep_gemm
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
    # Experiment loads the parent's exact existing Humming-layout converter.
    slab_dir = Path(__file__).resolve().parents[2] / "research_2026-11-18_rl/expert_slab"
    if str(slab_dir) not in sys.path:
        sys.path.insert(0, str(slab_dir))
    from slab import materialize
    assert x.is_cuda and torch.cuda.get_device_capability(x.device)[0] == 9
    assert x.dtype == out.dtype == torch.bfloat16 and ids.dtype in (torch.int32,torch.int64)
    assert x.is_contiguous() and out.is_contiguous() and ids.is_contiguous()
    t,h=x.shape
    assert ids.shape[0] == t and out.shape == (*ids.shape,h) and t>0
    assert ids.device == x.device == out.device
    k=ids.shape[1]
    total=layer.w13_post_scale.numel()
    assert layer.w13_post_scale.dtype == layer.w2_post_scale.dtype == torch.bfloat16
    assert total == layer.w2_post_scale.numel()
    weights=[]
    for stage,n in (("w13",2*h),("w2",h)):
        meta=layer.humming_metas[stage]
        assert not meta.should_apply_bs_on_c and not meta.use_fused_e8m0_scale
        assert meta.weight_scale_group_size==16
        w=getattr(layer,meta.weight_name)
        sf=getattr(layer,meta.weight_scale_name)
        assert w.device == sf.device == x.device and w.shape[0]==sf.shape[0]==total
        assert w.numel()*w.element_size()==total*n*h//2 and sf.numel()==total*n*h//16
        weights.append((w,sf,"wgmma" in str(meta.mma_type).lower()))
    alignment=deep_gemm.get_mk_alignment_for_contiguous_layout()
    if isinstance(alignment,tuple):
        alignment=alignment[0]
    size=min(slab_size,total)
    chunk=min(t,token_chunk)
    mem=memory_budget(h,size,chunk,k,alignment)
    if mem["bf16_total_bytes"] + (16<<20) > 1<<30:
        raise ValueError("Slab geometry exceeds the 1-GiB budget including routing headroom")
    cap=mem["padded_rows"]
    w13,w2,io,up,act,lut = (workspace.w13, workspace.w2, workspace.io,
                             workspace.up, workspace.act, workspace.lut)
    assert w13.shape == (size,2*h,h) and w2.shape == (size,h,h)
    assert io.shape[0] >= cap and io.shape[1] == h and act.shape == io.shape
    assert up.shape == (io.shape[0],2*h)
    assert all(z.device == x.device and z.is_contiguous() and z.dtype == torch.bfloat16
               for z in (w13,w2,io,up,act,lut))
    # data aliases the input/output slice only after W13 has read it on this stream.
    for first in range(0,total,size):
        group=min(size,total-first)
        expert=torch.arange(first,first+group,device=x.device,dtype=torch.int32)
        for stage,slab_buf in zip(weights,(w13,w2)):
            w,sf,transpose=stage
            materialize(w,sf,lut,expert,slab_buf[:group],transpose)
        for start in range(0,t,chunk):
            tokens=x[start:start+chunk]
            local_ids=ids[start:start+chunk]
            plan=routes(local_ids,first,group,alignment)
            r=plan.row_expert.numel()
            data=io[:r]
            gu=up[:r]
            hidden=act[:r]
            data.zero_()
            move(tokens,data,plan,k,True)
            native_consumer(data,w13[:group],gu,plan.group_ends)
            # Dense weights contain Humming's intrinsic block-scale /4 convention.
            # Reproduce the BF16 output, then post-scale, before the scored activation.
            gu.mul_(4)
            scale_active_rows(gu,plan.row_expert[:,None],layer.w13_post_scale)
            silu_and_mul(gu,hidden,expert_ids=plan.row_expert,expert_step=1)
            native_consumer(hidden,w2[:group],data,plan.group_ends)
            data.mul_(4)
            if not defer_downscale:
                scale_active_rows(data,plan.row_expert[:,None],layer.w2_post_scale)
            dest=out[start:start+chunk].view(-1,h)
            move(data,dest,plan,k,False)
    return out,ids

