"""Two-Gloo-rank fallback/collective protocol test. CPU only, no model weights needed."""
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from sglang_adapter import ProjectionRouter


class Group:
    world_size=2
    def __init__(self,rank):
        self.rank_in_group=rank
        self.cpu_group=dist.group.WORLD
        self.gathers=0
    def all_gather(self,x,dim):
        parts=[torch.empty_like(x) for _ in range(2)]
        dist.all_gather(parts,x)
        self.gathers+=1
        return torch.cat(parts,dim)


class FakePlan:
    def __init__(self,mode): self.mode=mode
    def close(self): pass
    def local(self,x,w,rank):
        if self.mode=="ineligible_at_runtime" and rank==1: return None
        y=F.linear(x,w).chunk(2,dim=1)[rank].contiguous()
        if self.mode=="one_bit" and rank==1: y.view(torch.int16)[0,0]^=1
        return y


def worker(rank,url):
    dist.init_process_group("gloo",rank=rank,world_size=2,init_method=url)
    torch.manual_seed(4017)
    nx=torch.nn.Linear(32,16,bias=False,dtype=torch.bfloat16)
    group=Group(rank)
    for mode in ["all_good","not_all_registered","one_bit","ineligible_at_runtime"]:
        model=ProjectionRouter(nx,group)
        v=torch.randn(8,32,dtype=torch.bfloat16)
        fresh=torch.randn(8,32,dtype=torch.bfloat16)
        cold=torch.randn(4,32,dtype=torch.bfloat16)
        def search(_examples,world):
            return (None if mode=="not_all_registered" and rank==1 else FakePlan(mode)),{}
        n=group.gathers
        model.register([v],discover_fn=search)
        assert torch.equal(model(cold).view(torch.int16),nx(cold).view(torch.int16))
        assert n==group.gathers
        assert torch.equal(model(fresh).view(torch.int16),nx(fresh).view(torch.int16))
        assert group.gathers==n+int(mode!="not_all_registered")
        model.close()
    dist.destroy_process_group()
    if rank==0: print("2-rank Gloo: startup disagreement, local rejection, one-bit corruption, cold-shape fallback and collective order passed")


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as td:
        mp.spawn(worker,args=((Path(td)/"torch.shared").as_uri(),),nprocs=2,join=True)

