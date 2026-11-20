"""Check independently reachable pages, padding, duplicates, shrink and grow, then compile for both cards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import triton
from snapshot import snapshot,_snapshot

here=Path(__file__).resolve().parent

def interpret():
    gen=torch.Generator().manual_seed(27441);checks=0;values=0
    for b,capacity,window,q in ((1,31,3,1),(4,255,67,8),(7,613,129,13),(17,1033,311,8)):
        # Row padding and a column-offset base exercise non-contiguous row strides.
        table=torch.randint(0,4900,(b+2,capacity+11),generator=gen,dtype=torch.int32)[:,3:3+capacity]
        bank=torch.randint(-1,12000,(4900,),generator=gen,dtype=torch.int64)
        outputs=[torch.full((b,capacity+23),-79123,dtype=torch.int32)[:,5:5+capacity] for _ in range(2)]
        sizes=torch.empty(b,dtype=torch.int32);cu=torch.empty(b+1,dtype=torch.int32)
        for i in range(3):
            seq=torch.randint(0,capacity+90,(b,),dtype=torch.int32,generator=gen)
            seq[0]=[0,capacity+32,9][i]
            ids=torch.randint(0,b+2,(b,),dtype=torch.int64,generator=gen)
            if b>2:ids[:2]=0
            before=[x.clone() for x in outputs]
            snapshot(sizes,cu,outputs[0],table,ids,capacity,seq,q,1,outputs[1],bank,swa_suffix_tokens=window+q,full_block=256,swa_block=32)
            end=seq+q
            assert torch.equal(sizes,end) and torch.equal(cu,torch.cat((torch.zeros(1),end.cumsum(0))).int())
            pos=torch.arange(capacity)[None,:]
            live=pos<end[:,None]
            mapped=table[ids]
            touch=live & (pos>=((end-window-q).clamp_min(0)//32*32)[:,None])
            assert torch.equal(outputs[0],torch.where(live,mapped,before[0]))
            assert torch.equal(outputs[1],torch.where(touch,bank[mapped.long()].int(),before[1]))
            for k in range(len(outputs)):assert torch.equal(outputs[k].clone(),outputs[k])
            table.random_(4900,generator=gen);bank.random_(-1,32000,generator=gen)
            checks+=1;values+=2*b*capacity
    return dict(replays=checks,compared_storage_values=values)

def compile_cuda():
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    types=["*i32","*i32","*i32","*i32","*i64","*i32","*i32","*i64"]
    out=[]
    for sm in (86,90):
        for block,warp in ((1024,4),(2048,4),(4096,4),(4096,8),(8192,8)):
            k=dict(STRIDE_SOURCE=131072,STRIDE_FULL=131072,STRIDE_SWA=131072,LIMIT=131072,
                   B=32,ROUND_B=32,DELTA=8,WINDOW=2055,FULL_BLOCK=block,SWA_BLOCK=256,FULL_TILES=131072//block)
            types0=dict(zip(_snapshot.arg_names,types));types0.update({n:"constexpr" for n in k})
            src=ASTSource(_snapshot,types0,k,{(i,):[["tt.divisibility",16]] for i in range(8)})
            asm=triton.compile(src,target=GPUTarget("cuda",sm,32),options={"num_warps":warp}).asm
            out.append(dict(sm=sm,block=block,warps=warp,cubin_bytes=len(asm["cubin"]),sha256=hashlib.sha256(asm["cubin"]).hexdigest()))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("interpret","compile"));mode=p.parse_args().mode
    result=interpret() if mode=="interpret" else compile_cuda()
    (here/(mode+".json")).write_text(json.dumps(result,indent=2))
    print(result if mode=="interpret" else f"{len(result)} compiled SM86 and SM90 cases")
