"""Time the local selector-to-metadata and acceptance part at equal eight-row budget."""
import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

H=Path(__file__).resolve().parent; R=H.parent
K=R/"research_2026-11-18_rl/lattice_tree"
sys.path.insert(0,str(K))
sys.path.insert(0,str(R/"research_2026-11-18_rl_prefill/decode_pages/global_rows"))
from real_layout import load as metadata_loader
from fixed_tree_adapter import TreeStep

def named(name,file):
    spec=importlib.util.spec_from_file_location(name,file)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def trial(b,length,rounds,seed,ordered):
    dev="cuda"; g=torch.Generator(device=dev).manual_seed(seed)
    cap=min(131072,1<<(length+8-1).bit_length());ragged=2047;width=8
    parent_meta=metadata_loader("parent_metadata")
    accept=named("fixed_tree_parent_accept",R/"sglang/python/sglang/kernels/ops/speculative/dflash.py")._compute_dflash_accept_bonus_triton_unchecked
    table=torch.empty((b+1,cap),device=dev,dtype=torch.int32)
    total=b*cap+1
    mapping=torch.arange(total,device=dev,dtype=torch.int32)
    # Fragmented page ids; each reserved suffix is owned by exactly one request.
    for i in range(b+1):
        table[i].copy_(torch.arange(cap,device=dev,dtype=torch.int32)+(max(i-1,0)*cap))
    swa=(mapping*17)%total
    req=torch.arange(1,b+1,device=dev,dtype=torch.int64)
    lens=torch.full((b,),length,device=dev,dtype=torch.int32)
    cols=torch.arange(8,device=dev)
    slots=torch.empty((b,8),device=dev,dtype=torch.int64)
    step=TreeStep(b,cap,dev,ragged)
    tables=(torch.empty_like(lens),torch.empty(b+1,device=dev,dtype=torch.int32),
            torch.empty((b,cap),device=dev,dtype=torch.int32),torch.empty((b,cap),device=dev,dtype=torch.int32))
    count=torch.empty_like(lens);a=torch.empty_like(lens)
    bonus=torch.empty(b,device=dev,dtype=torch.int64);end=torch.empty_like(bonus);out=torch.empty_like(slots)
    trace=np.load(K/"public_pair"/("trace_05.npz" if seed%2 else "trace_08.npz"))
    ids_seq=torch.as_tensor(trace["ids"],device=dev,dtype=torch.int64)
    score_seq=torch.as_tensor(trace["scores"],device=dev,dtype=torch.float32)
    token_seq=torch.as_tensor(trace["tokens"],device=dev,dtype=torch.int64)
    anchor_idx=torch.as_tensor(trace["anchors"],device=dev,dtype=torch.int64)
    chain=torch.empty_like(slots)
    target=torch.empty_like(slots)
    anchors=torch.empty_like(bonus);ids=torch.empty((b,7,16),device=dev,dtype=torch.int64)
    scores=torch.empty((b,7,16,16),device=dev,dtype=torch.float32);copied=torch.empty_like(scores)
    where=torch.arange(b,device=dev)
    def refresh(i):
        lens.copy_(length-(where+2*i)%max(1,min(17,length-1)))
        slots.copy_(table[req[:,None],lens[:,None]+cols])
        rows=(where+i*3)%(len(anchor_idx)-1)
        ids.copy_(ids_seq[rows]);scores.copy_(score_seq[rows])
        anchors.copy_(token_seq[anchor_idx[rows]])
        chain[:,0].copy_(anchors)
        prev=torch.zeros_like(where)
        for j in range(7):
            prev=scores[where,j,prev].argmax(-1)
            chain[:,j+1]=ids[where,j,prev]
        target.copy_(token_seq[anchor_idx[rows,None]+cols+1])
    def parent():
        ll,cu,pg,sw=tables
        parent_meta(ll,cu,pg,table,req,cap,lens,8,1,sw,swa)
    def run(arm):
        if arm=="tree":
            copied.copy_(scores)
            step.propose(ids,copied,anchors)
            step.stage(table,req,lens,slots,mapping,swa,ragged)
            parent()  # marked eager path still performs the scored native backend metadata init
            step.commit(target,table,req,lens,slots)
            torch.sub(step.counts,1,out=a)
        else:
            parent()
            accept(chain,target,a,count,bonus,out,lens,end)
    refresh(0)
    for arm in ("parent","tree","control"):run(arm)
    torch.cuda.synchronize();samples={k:[] for k in ("parent","tree","control")}
    for rr in range(rounds):
        refresh(rr)
        order=ordered[rr%len(ordered)]
        for arm in order:
            # Reset original immutable slot reservation before each timed arm.
            table[req[:,None],lens[:,None]+cols]=slots.to(torch.int32)
            start,end_ev=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();run(arm);end_ev.record();end_ev.synchronize()
            samples[arm].append(1000*start.elapsed_time(end_ev))
    return dict(median_us={k:statistics.median(v) for k,v in samples.items()},samples_us=samples,
                static_slab_bytes=sum(v.numel()*v.element_size() for k,v in vars(step).items() if torch.is_tensor(v) and k not in ("tokens","parents","depth","mask")),
                rows=b*width,context_cap=cap,peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True);p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--shapes",nargs="+",default=["1x2048","4x8192","16x74160","32x74160"])
    a=p.parse_args()
    order=[["parent","tree","control"],["control","tree","parent"],["tree","control","parent"],["parent","control","tree"],
           ["control","parent","tree"],["tree","parent","control"]]
    r=dict(seed=a.seed,device=torch.cuda.get_device_name(),torch=torch.__version__,order=order,
        scope="GPU event time of scored chain acceptance+request table metadata versus additional full tree proposal, true full and compact SWA page map and accepted virtual-slot permutation, plus scored request metadata init. Changed genuine public Qwen lattice with supplied node-order argmax for work cost. No native attention, model GEMM, TP, draft materialization, scheduler, Python host critical path or scored runtime.",
        parent_sha256=hashlib.sha256((R/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest(),shapes={})
    file=H/"fixed_tree"/f"gpu_components_{a.seed}.json"
    for shape in a.shapes:
        b,l=map(int,shape.split("x"));torch.cuda.reset_peak_memory_stats()
        r["shapes"][shape]=trial(b,l,a.rounds,a.seed,order)
        file.write_text(json.dumps(r,indent=2));print(shape,r["shapes"][shape]["median_us"],flush=True)
    print(file)
if __name__=="__main__":main()


