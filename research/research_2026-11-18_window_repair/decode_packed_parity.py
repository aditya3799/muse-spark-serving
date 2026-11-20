"""Candidate packing bit parity across eight independent vocab slices, with native PyTorch top-k."""
from pathlib import Path
import json,sys,torch
from decode_packed_kernels import pack_candidate_scores,pick_packed_candidates
H=Path(__file__).resolve().parent;D=H/"decode_selector"
def run(device):
    g=torch.Generator(device=device).manual_seed(987613)
    count=0;cases=0
    for dt in (torch.float32,torch.bfloat16,torch.float16):
        for b in (1,4,16,32):
            n=b*7;k=16;vals=[];ids=[];raw=[]
            for rank in range(8):
                v=torch.randn((n,k),generator=g,device=device).to(dt)
                i=torch.randint(0,25256,(n,k),generator=g,device=device).int()
                v[:,1:4]=v[:,:1].clone();v[::3,5]=-float("inf");v[1::3,7]=float("nan");v[2::3,12]=-0.
                if dt==torch.float32:v.view(torch.int32)[::4,9]=0x7fc00123
                offset=rank*25256
                vals.append(v.float());ids.append(i.long()+offset)
                raw.append(pack_candidate_scores(v,i,offset))
            allvals=torch.cat(vals,-1);allids=torch.cat(ids,-1)
            top,s=torch.topk(allvals,k,-1);gold=torch.gather(allids,-1,s)
            actual,v=pick_packed_candidates(torch.cat(raw,-1),k)
            assert torch.equal(actual,gold) and torch.equal(top.view(torch.int32),v.view(torch.int32)),(dt,b)
            count+=top.numel()+actual.numel();cases+=1
    # Exercise a 32-bit unsigned ID that would appear as negative if truncated through int32.
    x=torch.tensor([[1.,0.]],device=device);i=torch.tensor([[3,9]],device=device,dtype=torch.int64)
    raw=pack_candidate_scores(x,i,(1<<32)-15);idx,vals=pick_packed_candidates(raw,2)
    assert idx.tolist()==[[(1<<32)-12,(1<<32)-6]]
    (D/("packed_interpreter.json" if device=="cpu" else "packed_cuda_parity.json")).write_text(json.dumps(dict(device=device,cases=cases,exact_outputs=count,high_unsigned_ids=True),indent=2))
    print("candidate values and IDs passed",device,count)
if __name__=="__main__":run(sys.argv[1])

