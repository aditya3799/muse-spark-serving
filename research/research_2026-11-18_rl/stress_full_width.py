"""Probe selected full-width BF16 plans for reduction-order differences in finite inputs."""
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
OTHER=HERE.parent/"research_2026-11-18_lt_audit"
sys.path.insert(0,str(OTHER))
from audit_match import K,N,NAMES,load_lib,describe
from runtime_projection import native_workspace


def operands(m, seed, case, w):
    gen=torch.Generator().manual_seed(seed)
    rows=torch.arange(N)
    x=torch.ones((m,K),device="cuda",dtype=torch.bfloat16)
    exp=torch.randint(12,28,(N,),generator=gen)
    small=torch.full((N,),2.0**-24)
    large=torch.ldexp(small,exp)
    sgn=torch.where(rows%3==0,-1.,1.)
    w.zero_()
    if case in ("adjacent_tiles","far_tiles","split_edges"):
        if case=="adjacent_tiles":
            step=2**torch.randint(3,8,(N,),generator=gen)
            center=torch.randint(1,255,(N,),generator=gen)*128
            pos=torch.stack((center-1,center,center+step),dim=1)
        elif case=="far_tiles":
            band=K//4
            pos=torch.stack([torch.randint(i*band,(i+1)*band,(N,),generator=gen) for i in (0,1,3)],dim=1)
        else:
            # Place operands across possible split-K and tensor-core tile boundaries.
            splits=torch.randint(2,17,(N,),generator=gen)
            width=((K//splits)//16)*16
            center=torch.randint(1,64,(N,),generator=gen)% (splits-1)+1
            center=(width*center).clamp(32,K-32)
            pos=torch.stack((center-17,center-1,center+1),dim=1)
        vals=torch.stack((large,small,-large),dim=1)*sgn[:,None]
        perm=torch.rand(N,3,generator=gen).argsort(dim=1)
        w[rows[:,None].cuda(),pos.cuda()]=vals.gather(1,perm).to(device="cuda",dtype=w.dtype)
    else:
        # Reorder mirrored terms by chunks rather than adjacent half-blocks.
        column=torch.arange(K//2,device="cuda")
        base=((column%251).float()/251.-.5).bfloat16()
        w[:,:K//2]=base
        w[:,K//2:]=-base
        pick=torch.randint(0,K//2,(N,),generator=gen)
        w[rows.cuda(),pick.cuda()]=(w[rows.cuda(),pick.cuda()].float()+2.0**-9).bfloat16()
        x[:,1::7]=.5
        x[:,K//2:]=x[:,:K//2].clone()
    # Different rows include positive, negative and scaled activations.
    row_gain=torch.tensor([1.,-.5,2.,.25],device="cuda",dtype=x.dtype)[torch.arange(m,device="cuda")%4]
    x*=row_gain[:,None]
    return x


def main():
    details=json.loads((OTHER/"full_width_detail.json").read_text())
    _,ws=native_workspace()
    lib=load_lib()
    cap=max(ws,32*2**20)
    scratch=torch.empty(cap,device="cuda",dtype=torch.uint8)
    w=torch.empty((N,K),device="cuda",dtype=torch.bfloat16)
    result={}
    for m in (8,32,128,256,753):
        chosen=details[str(m)]["chosen"]
        if not chosen:continue
        plan=lib.make_sweep(m,N,N,K,N,ws,cap,details[str(m)]["max_split"])
        assert plan
        try:
            index=next(i for i in range(lib.sweep_count(plan)) if all(describe(lib,plan,i)[k]==chosen[k] for k in NAMES))
            out=torch.empty((m,N),device="cuda",dtype=torch.bfloat16)
            record=[]
            for j,case in enumerate(("adjacent_tiles","far_tiles","split_edges","dense_reordered")):
                x=operands(m,seed=160731+j,case=case,w=w)
                ref=F.linear(x,w)
                status=lib.sweep_call(plan,index,1,w.data_ptr(),x.data_ptr(),out.data_ptr(),scratch.data_ptr(),cap,
                                      torch.cuda.current_stream().cuda_stream)
                assert status==0
                bits=out.view(torch.int16)!=ref.view(torch.int16)
                row=dict(case=case,numeric_disagreements=int((out!=ref).sum()),bit_disagreements=int(bits.sum()),
                         values=ref.numel(),max_abs=float((out.float()-ref.float()).abs().max()))
                if row["numeric_disagreements"]:
                    r,c=(out!=ref).nonzero()[0].tolist()
                    row["first_difference"]=dict(row=r,column=c,parent=float(ref[r,c]),candidate=float(out[r,c]))
                record.append(row)
            result[str(m)]=record
            (HERE/"full_width_stress.json").write_text(json.dumps(result,indent=2)+"\n")
            print(m,[(r["case"],r["numeric_disagreements"]) for r in record],flush=True)
        finally:
            lib.drop_sweep(plan)


if __name__=="__main__":main()
