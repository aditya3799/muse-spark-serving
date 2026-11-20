"""Validate the one-program best-first selector against exhaustive or independent beam search."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import torch
from fused_tree import build
from selector_tree import build_tree
here=Path(__file__).resolve().parent

def paths(tokens,parents,depth,bits=None):
    result=[]
    for r in range(len(tokens)):
        rows=[()]
        masks=[1]
        for i in range(1,len(tokens[r])):
            p=int(parents[r,i])
            assert 0<=p<i
            rows.append(rows[p]+(int(tokens[r,i]),))
            masks.append(masks[p]|(1<<i))
            assert len(rows[-1])==int(depth[r,i])
            if bits is not None:assert int(bits[r,i])==masks[i]
        assert len(set(rows))==len(rows)
        result.append(set(rows[1:]))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
    if args.device=="cpu":assert os.environ.get("TRITON_INTERPRET")=="1"
    torch.set_num_threads(1)
    g=torch.Generator().manual_seed(51811);count=0;ties=0
    for k,d in ((4,3),(16,7)):
        ids=torch.arange(k*d).reshape(1,d,k).repeat(3,1,1).to(args.device)
        anchor=torch.tensor([777,888,999],device=args.device)
        raw=torch.randn(3,d,k,k,generator=g).to(args.device)
        mass=(torch.rand(3,d,generator=g)*.65+.32).to(args.device)
        for n in (5,8,12,16):
            for fork in (.5,1.):
                a=build(ids,raw,anchor,n,temperature=.65,after_fork=fork,log_mass=mass.log())
                b=build_tree(ids,raw/.65,anchor,n,mass[:,:,None].expand(-1,-1,k),after_fork=fork)
                aa=paths(*(v.cpu() for v in a))
                bb=paths(*(v.cpu() for v in b[:3]))
                assert aa==bb,(k,d,n,fork)
                count+=len(ids)
                if k==4:
                    edges=(raw/.65).log_softmax(-1)+mass.log()[:,:,None,None]
                    for r in range(3):
                        matches=[]
                        for l in range(1,d+1):
                            for seq in itertools.product(range(k),repeat=l):
                                value=0.;after=False
                                for j,t in enumerate(seq):
                                    prev=seq[j-1] if j else 0
                                    value+=float(edges[r,j,prev,t])+int(after)*math.log(fork)
                                    after|=t!=int(edges[r,j,prev].argmax())
                                matches.append((value,tuple(int(ids[r,j,t]) for j,t in enumerate(seq))))
                        assert aa[r]=={p for _,p in sorted(matches,reverse=True)[:n-1]}
        for val in (0.,float("-inf"),float("nan"),float("inf")):
            extreme=torch.full_like(raw,val)
            extreme[:,0,0,1]=float("inf")
            result=build(ids,extreme,anchor,16,after_fork=.5)
            paths(*(v.cpu() for v in result));ties+=len(ids)
    record=dict(device=args.device,matched_independent_trees=count,edge_trees=ties,
                conditions="Random logits, exhaustive small tree, two fork penalties, depths 3/7, K4/K16, zero/NaN/infinite and tied rows.")
    (here/("fused_tree_"+args.device+".json")).write_text(json.dumps(record,indent=2))
    print(record)

if __name__=="__main__":main()
