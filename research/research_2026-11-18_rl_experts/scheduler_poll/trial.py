"""CPU-only paired Gloo control path using pinned source, with the actual TP ring."""
import argparse
import contextlib
import hashlib
import json
import logging
from multiprocessing import shared_memory, resource_tracker
import os
from pathlib import Path
import pickle
import random
import statistics
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import uuid

os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

here=Path(__file__).resolve().parent
def module(name, **kw):
    mod=ModuleType(name)
    mod.__dict__.update(kw)
    sys.modules[name]=mod
    return mod

def make_shm_name(kind):
    return "okc_poll_%s_%s_%s" % (kind,os.getpid(),uuid.uuid4().hex[:8])

def source_func(filename,name,env):
    code=(here/filename).read_text()
    exec("from __future__ import annotations\n"+code,env)
    return env[name]

def unavailable(*args,**kw):
    raise AssertionError("Unexpected remote branch in single-host fixture")

module("sglang.srt.environ",envs=SimpleNamespace(SGLANG_PORT=SimpleNamespace(get=lambda:None)))
module("sglang.srt.utils.network",NetworkAddress=unavailable,get_local_ip_auto=unavailable,get_open_port=unavailable)
module("sglang.srt.utils.stale_shm_cleanup",make_shm_name=make_shm_name)
same=source_func("same_node.py","in_the_same_node_as",dict(torch=torch,shared_memory=shared_memory,
                 make_shm_name=make_shm_name,contextlib=contextlib,patch=patch,logger=logging.getLogger()))
module("sglang.srt.distributed.parallel_state",in_the_same_node_as=same)
from shm_reference import MessageQueue


def worker(rank,world,path,seed,rounds,nsteps):
    torch.set_num_threads(1)
    pginit="file://"+path
    dist.init_process_group("gloo",init_method=pginit,rank=rank,world_size=world)
    pg=dist.group.WORLD
    # The scored factory creates this ring for the TP CPU group with precisely these sizes.
    queue=MessageQueue.create_from_process_group(pg,1<<22,6)
    assert queue.handle.remote_subscribe_port is None and len(queue.handle.local_reader_ranks)==world-1
    group=SimpleNamespace(cpu_group=pg,rank=rank,ranks=list(range(world)),mq_broadcaster=queue)
    parallel=SimpleNamespace(attn_dp_enabled=False,tp_size=world,tp_group=group)
    original=source_func("baseline_broadcast.py","broadcast_pyobj",dict(torch=torch,dist=dist,np=np,pickle=pickle,
                                                                        is_musa=lambda:False))
    scope=dict(get_parallel=lambda:parallel,broadcast_pyobj=original)
    parent=source_func("parent_receive.py","_broadcast_reqs_across_ranks",dict(scope))
    candidate=source_func("candidate_receive.py","_broadcast_reqs_across_ranks",dict(scope))
    rng=random.Random(seed)
    salt=rng.randrange(1000,9999)
    # Arrays represent the rank-zero receiver's original inputs before the request dispatcher.
    # Dense long prompts, local aborts attached by the receiver, and ordering must survive.
    for has_queue in (True, False):
        group.mq_broadcaster=queue if has_queue else None
        for arr,local in [([],[]),([{"tokens":[(i*17+salt)%32000 for i in range(74160)],"id":salt}],[{"abort":"A"}]),
                          ([],[{"abort":"B"}]),([{"ids":[9,3]},b"ordered",None],[])]:
            expected=arr+local
            for entry in (parent,candidate):
                got=entry(None,arr if rank==0 else None,local if rank==0 else None)
                assert got==expected
    group.mq_broadcaster=queue
    sample=[{"rid":str(salt),"input_ids":[(salt+i*3)%16237 for i in range(753)]}]
    report={}
    # Both timed arms immediately enter the full CPU consensus carried by every HiCache scheduler step.
    # The MIN is retained even for a quiet step and uses four int64 controls, including a digest.
    def run(arm,pattern):
        dist.barrier()
        start=time.perf_counter_ns()
        for j,active in enumerate(pattern):
            args=sample if active else []
            got=arm(None,args if rank==0 else None,None)
            assert bool(got)==active
            if active:
                assert got[0]["rid"]==str(salt) and got[0]["input_ids"][-1]==sample[0]["input_ids"][-1]
            ctrl=torch.tensor([rank%3,(rank+j)%2,salt,-salt],dtype=torch.int64)
            dist.all_reduce(ctrl,op=dist.ReduceOp.MIN,group=pg)
            vals=ctrl.tolist()
            assert vals==[0,0,salt,-salt]
        elapsed=(time.perf_counter_ns()-start)/1000/len(pattern)
        upper=torch.tensor(elapsed,dtype=torch.float64)
        dist.all_reduce(upper,op=dist.ReduceOp.MAX)
        return upper.item()

    for label, fraction in (("idle_or_empty",0),("one_arrival_in_eight",0.125),("every_poll_has_arrival",1)):
        passes={"baseline":[],"candidate":[],"control":[]}
        for _ in range(2):
            run(parent,[False,True,False,False])
            run(candidate,[False,True,False,False])
        for i in range(rounds):
            pattern=[True]*round(fraction*nsteps)+[False]*(nsteps-round(fraction*nsteps))
            rng.shuffle(pattern)
            order=["baseline","candidate","control"] if (i+seed)%2==0 else ["candidate","baseline","control"]
            for entry in order:
                duration=run(candidate if entry=="candidate" else parent,pattern)
                passes[entry].append(duration)
        report[label]={"times_us_per_poll_including_min":passes,
            "median_ratio":statistics.median(a/b for a,b in zip(passes["baseline"],passes["candidate"])),
            "paired_median_us_saved":statistics.median(a-b for a,b in zip(passes["baseline"],passes["candidate"])),
            "aa_median_ratio":statistics.median(a/b for a,b in zip(passes["baseline"],passes["control"]))}
    if rank==0:
        target=here/("cpu_tp%d_seed%d.json"%(world,seed))
        target.write_text(json.dumps(dict(world=world,seed=seed,rounds=rounds,nsteps=nsteps,
            minimum_complete_payload_and_local_abort_parity=True,membership=queue.handle.local_reader_ranks,
            shared_memory_ring_bytes=6*(1<<22),cases=report),indent=2))
        print(target.name,[(n,round(d["median_ratio"],3),round(d["paired_median_us_saved"],1)) for n,d in report.items()],flush=True)
    dist.barrier()
    dist.destroy_process_group()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ranks",type=int,default=8)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--rounds",type=int,default=10)
    p.add_argument("--steps",type=int,default=16)
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        # Start the resource tracker once, before fork. Torch's weights stay shared by copy-on-write.
        resource_tracker.ensure_running()
        mp.start_processes(worker,args=(a.ranks,str(Path(tmp)/"store"),a.seed,a.rounds,a.steps),nprocs=a.ranks,
                           join=True,start_method="fork")
