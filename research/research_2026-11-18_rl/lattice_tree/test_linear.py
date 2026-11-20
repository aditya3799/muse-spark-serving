"""Verify the Bellman chain against exhaustive path utilities and the incumbent greedy choice."""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from linear_planner import _linear
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here/"public_pair"))
from bellman import policy

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
    if args.device=="cpu":assert os.environ.get("TRITON_INTERPRET")=="1"
    rng=np.random.default_rng(4957)
    checked=0; exhaustive=0
    for k,depth,b in ((2,3,16),(4,4,8),(16,7,16)):
        arr=rng.normal(size=(b,depth,k,k)).astype("float32")
        ids=np.broadcast_to(np.arange(depth*k,dtype=np.int64).reshape(1,depth,k),(b,depth,k)).copy()
        for temp,gamma in ((.5,.5),(1.,1.),(1.5,.75),(1.,0.)):
            choice=policy(arr,temp,gamma)
            gold=np.take_along_axis(ids,choice[:,:,None],axis=-1)[:,:,0]
            values=[torch.from_numpy(v).to(args.device) for v in (arr,ids)]
            dest=torch.empty((b,depth),device=args.device,dtype=torch.int64)
            _linear[(b,)](*values,dest,1/temp,gamma,depth,k,num_warps=4)
            np.testing.assert_array_equal(dest.cpu().numpy(),gold)
            checked+=b
            if k<=4 and gamma>0:
                for i in range(b):
                    probs=np.exp((arr[i]-arr[i].max(-1,keepdims=True))/temp)
                    probs/=probs.sum(-1,keepdims=True)
                    def reward(path):
                        prev=0;surv=1.;total=0.
                        for j,child in enumerate(path):
                            surv*=probs[j,prev,child];total+=(gamma**j)*surv;prev=child
                        return total
                    best=max(reward(alt) for alt in itertools.product(range(k),repeat=depth))
                    assert abs(best-reward(choice[i]))<1e-5
                    exhaustive+=1
    out=dict(device=args.device,checked=checked,independent_exhaustive_requests=exhaustive)
    (here/f"linear_{args.device}.json").write_text(json.dumps(out,indent=2)); print(out)
if __name__=="__main__":main()
