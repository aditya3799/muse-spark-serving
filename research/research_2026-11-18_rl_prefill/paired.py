"""Paired CUDA-graph timings and whole-buffer bitwise checks at the scored boundary."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import statistics
import sys
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
root = here.parent
sys.path.insert(0,str(root/"optimization_marlin_tiles_2026-11-15"))
from kernel_runtime import bootstrap
bootstrap(root/"sglang")

from sglang.kernels.ops.activation.activation import silu_and_mul
import route_activation as proposed

src = root/"release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py"
spec = importlib.util.spec_from_file_location("scored_w13_scale",src)
scored = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scored
spec.loader.exec_module(scored)
original_load = proposed.load_jit
def relocate(*args, **kwargs):
    assert kwargs["cuda_files"] == ["elementwise/route_activation.cuh"]
    kwargs["cuda_files"] = [str(here/"route_activation.cuh")]
    return original_load(*args, **kwargs)
proposed.load_jit = relocate

def digest(tensor):
    state=hashlib.sha256()
    for block in tensor.split(1024):
        state.update(block.view(torch.int16).cpu().numpy().tobytes())
    return state.hexdigest()

def execute(m,profile,seed,count):
    rng=random.Random(seed+m)
    generator=torch.Generator(device="cuda").manual_seed(seed+m)
    slots=m*8
    if profile=="uniform":
        global_ids=torch.randint(0,256,(m,8),generator=generator,device="cuda",dtype=torch.int32)
        ids=torch.where(global_ids<32,global_ids,-1).contiguous()
    else:
        # Cluster local work into runs, with variable occupancy and empty tokens.
        global_ids=torch.randint(0,256,(m,8),generator=generator,device="cuda",dtype=torch.int32)
        hot=(torch.arange(m,device="cuda")%128<16)[:,None]
        ids=torch.where((hot & (global_ids<160)) | (~hot & (global_ids<12)),global_ids%32,-1).contiguous()
    x=torch.empty((slots,8192),device="cuda",dtype=torch.bfloat16).normal_(generator=generator)
    y=torch.empty((slots,4096),device="cuda",dtype=torch.bfloat16)
    scales=torch.linspace(0.063,1.7,32,device="cuda",dtype=torch.float32).bfloat16()
    def parent():
        scored.scale_active_rows(x,ids,scales)
        silu_and_mul(x,y,expert_ids=ids.flatten(),expert_step=1)
    def candidate():
        proposed.scaled_prefill_activation(x,y,ids,scales)

    # A negative route must preserve its preexisting output.
    y.fill_(17)
    candidate()
    candidate_digest=digest(y)
    y.fill_(17)
    parent()
    reference_digest=digest(y)
    assert reference_digest==candidate_digest, (m,seed,reference_digest,candidate_digest)
    # Use unit scales only for the timed repeated input, because the scored scale
    # kernel writes its result in place. The numerical test above uses genuine scales.
    scales.fill_(1)
    graphs={}
    for name,fn in [("baseline",parent),("control",parent),("candidate",candidate)]:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name]=graph
    for _ in range(2):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    samples={name:[] for name in graphs}
    order=[]
    for trial in range(count):
        pair=["baseline","candidate"] if trial%2==0 else ["candidate","baseline"]
        pair.insert(rng.choice((0,2)),"control")
        order.append(pair.copy())
        for name in pair:
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/3)
    return {"weight":1,"times":samples,"order":order,"sha256_all_outputs":candidate_digest,
            "checked_bf16_values":y.numel(),"local_slot_count":int((ids>=0).sum()),
            "median_ratio":statistics.median(samples["baseline"])/statistics.median(samples["candidate"])}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pairs",type=int,default=9)
    parser.add_argument("--sizes",type=int,nargs="+",default=[753,3012,16384])
    args=parser.parse_args()
    parent_sha=hashlib.sha256((root/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()
    assert parent_sha=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    calls=[]
    actual_module=proposed.route_activation_module()
    class Counter:
        def run(self,*tensors):
            calls.append(tensors[0].shape[0])
            actual_module.run(*tensors)
    proposed.route_activation_module=lambda:Counter()
    report={"agent":"prefill","candidate":"token_group_w13_scale_silu","scope":"W13 BF16 scaling and filtered SiLU boundary, RTX 3050 SM86",
        "baseline_sha256":parent_sha,"candidate_sha256":hashlib.sha256((here/"candidate.diff").read_bytes()).hexdigest(),
        "correct":False,"active":False,"seed":args.seed,"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "unit":"us","bench":{"instances":1,"shapes":{}}}
    def save():
        args.out.write_text(json.dumps(report,indent=2)+"\n")
    save()
    for rows in args.sizes:
        profile="clustered" if rows==3012 else "uniform"
        name=f"M{rows}_{profile}"
        report["bench"]["shapes"][name]=execute(rows,profile,args.seed,args.pairs)
        report["active"]=len(calls)>0
        save()
        print(name,report["bench"]["shapes"][name]["median_ratio"],flush=True)
        torch.cuda.empty_cache()
    assert all(8*m in calls for m in args.sizes)
    report["correct"]=True
    report["launches_recorded_for_rows"]=sorted(set(calls))
    save()

