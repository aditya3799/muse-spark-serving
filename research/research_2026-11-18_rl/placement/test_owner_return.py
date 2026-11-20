"""Cross-rank return tests, including the exact original BF16 rank reductions."""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from placement import NATIVE, demand, draw, select, route_counts
from owner_return import return_rows

HERE=Path(__file__).resolve().parent


def run(rank,path,folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=path,rank=rank,world_size=8)
    results=[]
    for seed in (80717,83111):
        heat=demand(seed,"skewed")
        placement,swaps=select(route_counts(draw(seed+41,2048,heat)),4)
        for pattern in ("random","skewed","all_remote","fixed"):
            m,h=37,256
            ids=torch.from_numpy(draw(seed+413,m,heat if pattern=="skewed" else np.zeros(256)))
            if pattern=="all_remote":ids.fill_(-1)
            slots=NATIVE if pattern=="fixed" else placement
            gen=torch.Generator().manual_seed(seed+831)
            truth=torch.randn(m,8,h,generator=gen).to(torch.bfloat16)
            sc=torch.rand(m,8,generator=gen);sc/=sc.sum(-1,keepdim=True)
            norm=torch.linspace(.7,1.1,h,dtype=torch.bfloat16)
            scales=torch.linspace(.2,.6,256,dtype=torch.bfloat16)
            original=(ids>=32*rank)&(ids<32*(rank+1))
            physical=(ids[:,:,None]==torch.from_numpy(slots[rank])).any(-1)
            def finish(data,live):
                after=(data*scales[ids.clamp_min(0).long()][:,:,None]).to(torch.bfloat16)
                v=after.float()*norm
                v=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-8)*sc[:,:,None]
                v=torch.where(live[:,:,None],v,0.)
                out=torch.zeros(m,h)
                for j in range(8):out+=v[:,j]
                return out.to(torch.bfloat16)
            # Poison every unavailable raw row, so missing receptions cannot pass.
            raw=torch.where(physical[:,:,None],truth,torch.tensor(float("nan"),dtype=torch.bfloat16)).contiguous()
            naive=finish(raw,physical)
            wanted=finish(truth,original)
            got,local,sent=return_rows(raw,ids,slots,dist.group.WORLD)
            assert torch.equal(got[original],truth[original])
            assert torch.equal(local>=0,original)
            exact=finish(got,local>=0)
            assert torch.equal(wanted,exact)
            for arm in (naive,wanted,exact):dist.all_reduce(arm)
            assert torch.equal(wanted,exact) and torch.isfinite(exact).all()
            payload=torch.tensor(sent);dist.all_reduce(payload)
            results.append(dict(seed=seed,pattern=pattern,actual_p2p_bytes=int(payload),unchanged_after_restore=bool(torch.equal(wanted,exact)),
                                naive_changed_values=int((naive!=wanted).sum()),output_values=exact.numel()))
    if rank==0:Path(folder,"result.json").write_text(json.dumps(results))
    dist.destroy_process_group()


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(run,args=(Path(tmp,"init").as_uri(),tmp),nprocs=8,start_method="fork")
        rows=json.loads(Path(tmp,"result.json").read_text())
    assert any(r["naive_changed_values"] for r in rows) and any(r["actual_p2p_bytes"] for r in rows)
    (HERE/"owner_return.json").write_text(json.dumps(dict(scope="Actual 8-rank Gloo transfer; 256 hidden features and synthetic BF16 expert outputs. Measures preservation of original EP summation, not H200 time.",cases=rows),indent=2))
    print("8 ranks",len(rows),"cases",sum(r["output_values"] for r in rows),"bits-comparable BF16 values")
