"""Exercise actual route packing and scatter in the Triton CPU interpreter."""
import json
import os
from pathlib import Path
import torch
from consumer import memory_budget, move, routes

HERE=Path(__file__).resolve().parent
assert os.environ.get("TRITON_INTERPRET")=="1"
torch.set_num_threads(1)

def phase(a,b,d,p,align):
    last=0
    for g,end in enumerate(p.group_ends.tolist()):
        start=(last+align-1)//align*align
        assert end>=start
        if end>start:
            d[start:end].copy_((a[start:end].float()@b[g].float().T).bfloat16())
        last=end

def case(seed,ids,align):
    gen=torch.Generator().manual_seed(seed)
    t,k=ids.shape; h=64; e=7; gmax=3
    x=(torch.randn((t,h),generator=gen)*.5).bfloat16()
    w13=(torch.randn((e,2*h,h),generator=gen)*.4).bfloat16()
    w2=(torch.randn((e,h,h),generator=gen)*.4).bfloat16()
    s1=torch.linspace(.29,1.17,e).bfloat16()
    s2=torch.linspace(1.23,.71,e).bfloat16()
    results={}
    for defer in (False,True):
        gold=torch.full((t*k,h),-177.,dtype=torch.bfloat16)
        got=gold.clone()
        for first in range(0,e,gmax):
            g=min(gmax,e-first);p=routes(ids,first,g,align);n=p.row_expert.numel()
            slots=p.source_slot[p.dense_row>=0]; dest=p.dense_row[p.dense_row>=0]
            assert torch.unique(slots).numel()==slots.numel()==torch.unique(dest).numel()
            assert torch.equal(p.row_expert[dest.long()],ids.flatten()[slots.long()].int())
            assert ((dest>=0)&(dest<n)).all()
            a=torch.zeros((n,h),dtype=torch.bfloat16);up=torch.zeros((n,2*h),dtype=a.dtype)
            move(x,a,p,k,True)
            phase(a,w13[first:first+g],up,p,align)
            ix=p.row_expert.long().clamp_min(0)
            up.mul_(4); up.mul_(s1[ix,None])
            gate,value=up.chunk(2,dim=-1)
            h1=(torch.nn.functional.silu(gate.float())*value.float()).bfloat16()
            phase(h1,w2[first:first+g],a,p,align)
            a.mul_(4)
            if not defer:
                a.mul_(s2[ix,None])
            move(a,got,p,k,False)
        # Separate direct route reference with a scalar FP64 summation so compact
        # CPU GEMM shape choices cannot accidentally mask address/scale mistakes.
        for g in range(e):
            slots=torch.where(ids.flatten()==g)[0]
            if not len(slots):continue
            v=(x[slots//k].double()@w13[g].double().T).bfloat16().mul_(4).mul_(s1[g])
            gate,value=v.chunk(2,dim=-1);hidden=(torch.nn.functional.silu(gate.float())*value.float()).bfloat16()
            z=(hidden.double()@w2[g].double().T).bfloat16().mul_(4)
            if not defer:z.mul_(s2[g])
            gold[slots]=z
        # Random BF16 dot near a rounding tie may differ by 1 BF16 ULP across
        # FP32 vs FP64 reduction. Use a shared CPU FP32 GEMM separately below.
        mask=(got.view(torch.int16)!=gold.view(torch.int16))
        results[str(defer)]=dict(fp64_ref_differences=int(mask.sum()),max_abs=float((got.float()-gold.float()).abs().max()))
        assert torch.equal(got[(ids.flatten()<0)|(ids.flatten()>=e)],gold[(ids.flatten()<0)|(ids.flatten()>=e)])
        ref=torch.full_like(gold,-177)
        for g in range(e):
            slots=torch.where(ids.flatten()==g)[0]
            if not len(slots):continue
            v=(x[slots//k].float()@w13[g].float().T).bfloat16().mul_(4).mul_(s1[g])
            gate,value=v.chunk(2,dim=-1)
            hidden=(torch.nn.functional.silu(gate.float())*value.float()).bfloat16()
            z=(hidden.float()@w2[g].float().T).bfloat16().mul_(4)
            if not defer:z.mul_(s2[g])
            ref[slots]=z
        assert torch.equal(got.view(torch.int16),ref.view(torch.int16)),(seed,defer)
    return results

report={}
for seed in (3217,9127):
    gen=torch.Generator().manual_seed(seed)
    for name,ids in {
        "clustered":torch.randint(2,(35,4),generator=gen,dtype=torch.int32),
        "mixed":torch.randint(-3,10,(17,8),generator=gen,dtype=torch.int32),
        "empty":torch.full((3,8),-1,dtype=torch.int32),
        "duplicate":torch.tensor([[6,6,6,-1],[1,6,1,4],[2,0,3,2]],dtype=torch.int32),
        "singleton":torch.tensor([[4]],dtype=torch.int32)
    }.items():
        report[f"{seed}_{name}"]=case(seed,ids,16)
penalties={}
for a in [64,128,256]:
    penalties[a]=memory_budget(4096,8,512,8,a)
x=torch.arange(1,2048,dtype=torch.float32)*2**-134
order=int(((4*x).bfloat16().view(torch.int16)!=(4*x.bfloat16()).view(torch.int16)).sum())
out=dict(receipts=report,scratch=penalties,tiny_result_rounding_counterexamples=order,
         scope="Actual Torch routing and Triton CPU interpreter gather/scatter, CPU BF16 products and independent ungrouped row reference. Excludes native Hopper execution and parent Humming bit parity.")
(HERE/"cpu.json").write_text(json.dumps(out,indent=2))
print("CPU",len(report),"cases, 2 downscale modes",penalties[128],"underflow examples",order)

