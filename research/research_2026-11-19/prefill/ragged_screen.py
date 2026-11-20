"""Lossless screen of unpadded full-K row shards for nondivisible prefill projections.

All eight rank outputs are reconstructed on the single local device before any
timing claim. Timings are rank-local compute only; no collective is simulated.
"""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHA = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"


def ranges(m, world, mode="balanced"):
    q, r = divmod(m, world)
    if mode == "tail":
        cap = q + int(r != 0)
        return [(min(rank*cap,m), min((rank+1)*cap,m)) for rank in range(world)]
    return [(rank*q + min(rank, r), (rank+1)*q + min(rank+1, r)) for rank in range(world)]


def compare(a, b):
    d = a.float()-b.float()
    mismatches = int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
    return {"mismatches": mismatches, "elements": a.numel(), "max_abs": d.abs().max().item(),
            "relative_l2": (d.norm()/b.float().norm().clamp_min(1e-20)).item()}


def elapsed(fn):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    output = fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)*1000, output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[513, 753, 3012, 16383])
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--partition", choices=["balanced","tail"], default="balanced")
    args = p.parse_args()
    assert args.rounds % 2 == 0
    assert hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest() == SHA
    torch.set_num_threads(1)
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    report = {"seed": args.seed, "baseline_sha256": SHA, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "complete": False,
              "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "scope": "Full unrounded BF16 pool/unpool; all output rows reconstructed; measured rank legs exclude TP8 transport",
              "partition": args.partition, "cases": [], "stated_limits": ["synthetic inputs and weights", "Ampere not Hopper", "no true TP8 transport"]}
    out = HERE/args.out
    def save():
        out.write_text(json.dumps(report, indent=2)+"\n")
    save()
    for m in args.rows:
        for name,n,k in (("pool",4096,8192),("unpool",8192,4096)):
            for style in ("random", "cancellation", "magnitude", "integer"):
                x = torch.randn((m,k), device="cuda", dtype=torch.bfloat16, generator=gen)
                w = torch.randn((n,k), device="cuda", dtype=torch.bfloat16, generator=gen)/k**.5
                if style == "cancellation":
                    w[:,1::2] = -w[:,::2]
                    x[:,1::2] = x[:,::2]
                    x[:,-1] += .00390625
                elif style == "magnitude":
                    factors = torch.tensor([.015625,.125,1.,8.,64.],device="cuda",dtype=torch.bfloat16)
                    x *= factors[torch.arange(k,device="cuda")%5]
                elif style == "integer":
                    x = torch.randint(-2,3,(m,k),device="cuda",generator=gen).bfloat16()
                    w = torch.randint(-2,3,(n,k),device="cuda",generator=gen).bfloat16()*.03125
                ref = F.linear(x,w)
                cases = {"native_bf16":[], "fp32_then_bf16":[]}
                pair_ranges = ranges(m,8,args.partition)
                for start,end in pair_ranges:
                    cases["native_bf16"].append(F.linear(x[start:end],w))
                    cases["fp32_then_bf16"].append(torch.mm(x[start:end],w.T,out_dtype=torch.float32).bfloat16())
                item = {"name":name,"rows":m,"width":n,"k":k,"style":style,
                        "rank_rows":[end-start for start,end in pair_ranges],
                        "comparison":{variant:compare(torch.cat(parts),ref) for variant,parts in cases.items()}}
                # Equal-size all-gather sends ceil(M/8) rows/rank. Only the final
                # row of shorter ranks is padding; gather-consumer removes it.
                cap=(m+7)//8
                gathered=torch.zeros((8,cap,n),device="cuda",dtype=x.dtype)
                for rank,part in enumerate(cases["native_bf16"]):
                    gathered[rank,:part.shape[0]].copy_(part)
                rebuilt=torch.cat([gathered[r,:e-s] for r,(s,e) in enumerate(pair_ranges)])
                item["packed_gather_reconstruction"]=compare(rebuilt,torch.cat(cases["native_bf16"]))
                item["allgather_send_bytes_per_rank_ring_model"]=7*cap*n*2
                item["gather_staging_bytes"]=8*cap*n*2
                if style == "random":
                    times={"baseline":[],"control":[],"native_bf16_rank_max":[],"fp32_then_bf16_rank_max":[],"serial_reconstruction":[]}
                    for _ in range(2):
                        F.linear(x,w)
                        for start,end in pair_ranges:
                            F.linear(x[start:end],w)
                    for iteration in range(args.rounds):
                        labels=list(times)
                        if iteration%2: labels.reverse()
                        for label in labels:
                            if label in ("baseline","control"):
                                t,y=elapsed(lambda:F.linear(x,w))
                            elif label=="serial_reconstruction":
                                t,y=elapsed(lambda:torch.cat([F.linear(x[s:e],w) for s,e in pair_ranges]))
                            else:
                                measurements=[]
                                for start,end in pair_ranges:
                                    fn=(lambda start=start,end=end:F.linear(x[start:end],w)) if label.startswith("native") else (lambda start=start,end=end:torch.mm(x[start:end],w.T,out_dtype=torch.float32).bfloat16())
                                    tr,y=elapsed(fn)
                                    measurements.append(tr)
                                t=max(measurements)
                            times[label].append(t)
                    item["timing_us"]=times
                    item["timing_note"]="Rank max is the maximum of eight separately measured local legs, not distributed elapsed time."
                    item["median_us"]={label:statistics.median(v) for label,v in times.items()}
                    # Capture in an owned stream and perturb live rows between replays.
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        joined=torch.cat([F.linear(x[s:e],w) for s,e in pair_ranges])
                    x.neg_()
                    graph.replay()
                    item["changed_input_graph"]=compare(joined,F.linear(x,w))
                    del graph,joined
                report["cases"].append(item)
                save()
                print(m,name,style,{v:d["mismatches"] for v,d in item["comparison"].items()},flush=True)
                del x,w,ref,cases,gathered,rebuilt
    report["complete"]=True
    report["correct"]=all(c["comparison"]["native_bf16"]["mismatches"]==0 for c in report["cases"])
    report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    save()

if __name__=="__main__":
    main()
