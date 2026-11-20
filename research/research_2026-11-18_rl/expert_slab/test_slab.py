"""Check the inverse map for HMMA and WGMMA against the incumbent packer's forward indexing."""
import argparse
import ast
import json
import os
from pathlib import Path
import numpy as np
import torch
from slab import make_table,materialize

here=Path(__file__).resolve().parent
root=here.parents[1]
vendor=root/"swarm_2026-11-16/moe/humming_vendor/humming"

def scales(x):
    p=vendor/"transform.py"
    node=next(n for n in ast.parse(p.read_text()).body if getattr(n,"name",None)=="transform_humming_weight_scale")
    scope={"torch":torch}
    exec(compile(ast.Module([node],[]),str(p),"exec"),scope)
    return scope[node.name](x)

def pack(codes,wgmma):
    # Direct forward ownership loop, following process.cuh's weight_repack_nk and humming_pack_weight.
    e,n,k=codes.shape
    out=np.empty((e,k//16,2*n),dtype=np.uint32)
    for ei in range(e):
        for nb in range(n//64):
            for kb in range(k//16):
                for lane in range(32):
                    tmp=np.empty((4,2,2,2),dtype=np.uint32)
                    for i in range(8):
                        nn=nb*64+i*8+lane//4
                        for j in range(2):
                            start=kb*16+j*8+(lane%4)*2
                            if wgmma:tmp[i//2,j,i%2,:]=codes[ei,nn,start:start+2]
                            else:tmp[i//2,i%2,j,:]=codes[ei,nn,start:start+2]
                    order=[0,2,4,6,1,3,5,7]
                    words=np.sum(tmp.reshape(4,8)[:,order]<<(np.arange(8,dtype=np.uint32)*4),axis=1,dtype=np.uint32)
                    out[ei,kb,nb*128+lane*4:nb*128+lane*4+4]=words
    return torch.from_numpy(out.view(np.int32).copy())

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
    if args.device=="cpu":assert os.environ.get("TRITON_INTERPRET")=="1"
    rng=np.random.default_rng(7215);checked=0
    for e,n,k in ((3,128,192),(2,192,128)):
        codes=rng.integers(0,16,(e,n,k),dtype=np.uint8)
        sf=torch.from_numpy(rng.integers(0,256,(e,n,k//16),dtype=np.uint8))
        sf[sf==127]=0;sf[sf==255]=128
        lut=make_table("cpu")
        expected=lut[sf.long().repeat_interleave(16,dim=2),torch.from_numpy(codes).long()]
        ps=scales(sf).to(args.device)
        for wg in (False,True):
            pw=pack(codes,wg).to(args.device)
            for chosen in ([e-1,0],[1]):
                ids=torch.tensor(chosen,device=args.device,dtype=torch.int32)
                actual=torch.empty((len(chosen),n,k),device=args.device,dtype=torch.bfloat16)
                materialize(pw,ps,lut.to(args.device),ids,actual,wg)
                gold=expected[chosen].view(torch.int16)
                obs=actual.cpu().view(torch.int16)
                assert torch.equal(obs,gold),(wg,int((obs!=gold).sum()))
                checked+=obs.numel()
    r=dict(exact_BF16_values_including_signed_zero=checked,device=args.device,
           geometry="64x16 native pack tiles; both mini-block transpositions and full group-16 scale transformation")
    (here/("layout_"+args.device+".json")).write_text(json.dumps(r,indent=2))
    print(r)
if __name__=="__main__":main()
