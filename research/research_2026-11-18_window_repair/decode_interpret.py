"""Run both pinned and direct Triton kernels on CPU with Triton's official interpreter."""
from pathlib import Path
import json,sys,torch
H=Path(__file__).resolve().parent;D=H/"decode_selector";sys.path.insert(0,str(D))
import direct_walk,original_walk
def main():
    g=torch.Generator().manual_seed(11171);cnt=0
    for b in (1,4,16,32):
        for mode in ("greedy","mixed","sample"):
            ids=torch.randint(0,202048,(b,7,16),generator=g)
            scores=torch.randn((b,7,16,16),generator=g)
            mask=torch.ones(b,dtype=torch.bool)
            if mode=="mixed":mask[::2]=False
            if mode=="sample":mask.zero_()
            temp=torch.rand(b,generator=g)+.01;uniform=torch.rand(b,7,generator=g)
            if mode=="greedy":
                scores[:,0,0,2:4]=8;scores[0,1,2,6]=float("nan")
            args=dict(candidate_ids=ids,scores=scores,greedy_mask=mask,temperatures=temp,uniforms=uniform)
            x,y=original_walk.selector_walk_triton(**args)
            a=torch.full_like(x,-4);z=torch.full_like(y,-9);out=torch.full_like(ids,-3)
            direct_walk.selector_walk_triton(**args,tokens=a,q_rows=z,candidate_out=out)
            assert torch.equal(x,a) and torch.equal(y.view(torch.int32),z.view(torch.int32)) and torch.equal(out,ids)
            cnt+=x.numel()+y.numel()+out.numel()
    (D/"interpreter.json").write_text(json.dumps(dict(cases=12,compared_entries=cnt,correct=True,seed=11171,scope="CPU official Triton interpreter; NaNs, ties, changing mixed/risk rows"),indent=2))
    print("passed",cnt)
if __name__=="__main__":main()

