"""Exercise actual Triton kernels via CPU interpreter and the independent frozen forward packer."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from group_prep import convert, gather, route, route_buffers, scatter
folder=Path(__file__).resolve().parent
sys.path.insert(0,str(folder.parents[1]/"research_2026-11-18_rl/expert_slab"))
from test_slab import pack,scales


def test_pack(rng):
    checked=0
    for e,n,k in ((3,128,192),(2,192,128)):
        q=rng.integers(0,16,(e,n,k),dtype=np.uint8)
        sf=rng.integers(0,256,(e,n,k//16),dtype=np.uint8)
        sf[:,:,:4]=np.arange(e*n*4,dtype=np.uint32).astype(np.uint8).reshape(e,n,4)
        sh=scales(torch.from_numpy(sf))
        chosen=[e-1,0]
        expected_q=q[chosen,:,::2]|(q[chosen,:,1::2]<<4)
        expected_s=sf[chosen].transpose(0,2,1).reshape(len(chosen),k//64,4,n).astype(np.uint32)
        expected_s=np.bitwise_or.reduce(expected_s<<(np.arange(4,dtype=np.uint32)[None,None,:,None]*8),axis=2)
        storage=torch.full((len(chosen)*n*k*9//16+32,),0x55,dtype=torch.uint8)
        for wg in (False,True):
            pw=pack(q,wg)
            a,b=convert(pw,sh,torch.tensor(chosen,dtype=torch.int32),storage,n,k,wg)
            np.testing.assert_array_equal(expected_q,a.numpy())
            np.testing.assert_array_equal(expected_s,b.numpy().view(np.uint32))
            assert torch.all(storage[-32:]==0x55)
            checked+=a.numel()*2+b.numel()*4
    return checked


def test_routes(rng):
    e=5
    total=0
    for m in (1,37,67):
        ids=torch.empty((m,8),dtype=torch.int32)
        bufs=route_buffers(ids,e)
        h=64
        x=torch.arange(m*h,dtype=torch.float32).reshape(m,h).bfloat16()
        grouped=torch.empty((m*8+e*16,h),dtype=torch.bfloat16)
        output=torch.empty((m*8,h),dtype=torch.bfloat16)
        for mode in ("mixed","none","skew","all"):
            if mode=="mixed":
                key=rng.integers(-3,e+2,(m,8),dtype=np.int32)
            elif mode=="none":
                key=np.full((m,8),-1,dtype=np.int32)
            elif mode=="skew":
                key=rng.choice(np.array([-1,0,e-1],dtype=np.int32),(m,8),p=[.3,.6,.1])
            else:
                key=np.full((m,8),e-1,dtype=np.int32)
            ids.copy_(torch.from_numpy(key))
            mapping=route(ids,bufs)
            dest,base,counts,pad=mapping
            canonical=key.flatten()
            gold=np.full(m*8,-1,dtype=np.int32)
            offset=0
            grouped.fill_(float("nan"))
            gather(x,grouped,mapping)
            for j in range(e):
                slots=np.flatnonzero(canonical==j)
                length=max(16,(len(slots)+15)//16*16)
                assert (int(base[j]),int(counts[j]),int(pad[j]))==(offset,len(slots),length)
                gold[slots]=np.arange(offset,offset+len(slots))
                assert torch.equal(grouped[offset:offset+len(slots)],x[slots//8])
                assert not grouped[offset+len(slots):offset+length].count_nonzero()
                offset+=length
            assert grouped[offset:].isnan().all()
            np.testing.assert_array_equal(dest.numpy(),gold)
            scatter(grouped,output,dest)
            expected=x.repeat_interleave(8,0).mul_(4)
            expected[(canonical<0)|(canonical>=e)]=0
            assert torch.equal(output,expected)
            total+=output.numel()
    return total


if __name__=="__main__":
    assert os.environ.get("TRITON_INTERPRET")=="1"
    receipts=[]
    for seed in (77153,91407):
        rng=np.random.default_rng(seed)
        receipts.append(dict(seed=seed,codes_and_scale_bytes=test_pack(rng),raw_scatter_BF16=test_routes(rng)))
    out=dict(device="CPU Triton interpreter",receipts=receipts)
    (folder/"group_prep_checked.json").write_text(json.dumps(out,indent=2))
    print(out)
