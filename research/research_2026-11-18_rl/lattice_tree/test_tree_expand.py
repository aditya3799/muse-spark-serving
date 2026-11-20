"""Check compact branch metadata against the pinned FA3 expand path's dense-mask sort."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from tree_expand import _expand


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    dev=parser.parse_args().device
    if dev=="cpu": assert os.environ.get("TRITON_INTERPRET")=="1"
    rng=np.random.default_rng(71141)
    checked=0
    for n in (8,16,32):
        b=8
        for trial in range(6):
            parents=np.array([[0]+[rng.integers(j) for j in range(1,n)] for _ in range(b)])
            bits=np.ones((b,n),dtype=np.uint32)
            depths=np.zeros((b,n),dtype=np.int32)
            for i in range(1,n):
                bits[:,i]=bits[np.arange(b),parents[:,i]] | np.uint32(1<<i)
                depths[:,i]=depths[np.arange(b),parents[:,i]]+1
            seq=np.array([1,31,2049,31991,74160,131021,50,1024],dtype=np.int32)
            rng.shuffle(seq)
            slots=rng.choice(1893608+1,size=b*n,replace=False).reshape(b,n).astype(np.int32)
            def tensor(x): return torch.from_numpy(x).to(dev)
            compact=torch.empty((b*n,n),dtype=torch.int32,device=dev)
            counts=torch.empty(b*n,dtype=torch.int32,device=dev)
            total=torch.empty_like(counts)
            pos=torch.empty(b*n,dtype=torch.int64,device=dev)
            _expand[(b*n,)](tensor(slots),tensor(bits),tensor(depths),tensor(seq),
                            compact,counts,total,pos,n,num_warps=1)
            c=np.arange(n)[None,None,:]
            mask=(bits[:,:,None]>>c)&1
            order=np.argsort(np.where(mask,c,c+n),axis=-1,kind="stable")
            dense=np.take_along_axis(np.broadcast_to(slots[:,None,:],(b,n,n)),order,axis=2)
            np.testing.assert_array_equal(compact.cpu().numpy().reshape(b,n,n),dense)
            np.testing.assert_array_equal(counts.cpu().numpy().reshape(b,n),mask.sum(-1))
            np.testing.assert_array_equal(total.cpu().numpy().reshape(b,n),mask.sum(-1)+seq[:,None])
            np.testing.assert_array_equal(pos.cpu().numpy().reshape(b,n),depths+seq[:,None])
            checked+=b*n
    result=dict(device=dev,verified_node_rows=checked,scored_int64_allocator_capacity=1893608)
    print(result)
    Path(__file__).with_name("tree_expand_"+dev+".json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
