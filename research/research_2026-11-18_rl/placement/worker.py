"""Measure both placements on each of eight sequentially emulated ranks with actual scored Humming kernels."""
import argparse
import ast
import copy
import faulthandler
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from placement import NATIVE, demand, draw, local_map, route_counts, select

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"research_2026-11-18_target_moe"))
import bench_experts as fixture
for path in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",ROOT/"optimization_marlin_tiles_2026-11-15"):
    sys.path.insert(0,str(path))
from kernel_runtime import bootstrap
bootstrap(ROOT/"sglang")
import torch
from fixture_tuning_cache import cached_threshold_for_preparation

BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
MOE=fixture.SOURCE/"kernels/ops/moe"
fixture.module("sglang.kernels.ops.moe.muse_spark_humming_scale",MOE/"muse_spark_humming_scale.py")
fixture.module("sglang.kernels.ops.moe.muse_spark_humming_split",MOE/"muse_spark_humming_split.py")
finish=fixture.module("placement_finish",MOE/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
source=fixture.SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
names={"_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs"}
selected=[fn for fn in ast.parse(source.read_bytes()).body if getattr(fn,"name",None) in names]
assert len(selected)==3
native={"torch":torch}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),"exec"),native)

def banks(seed):
    torch.manual_seed(seed)
    raw=torch.nn.Module()
    for tag,n in [("w13",8192),("w2",4096)]:
        weight=torch.randint(256,(32,n,2048),device="cuda",dtype=torch.uint8)
        scale=torch.randint(40,68,(32,n,256),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)
        global_scale=torch.linspace(.125,.48,32,device="cuda",dtype=torch.bfloat16)
        for key,val in [("weight",weight),("weight_scale",scale),("weight_scale_2",global_scale)]:
            raw.register_parameter(tag+"_"+key,torch.nn.Parameter(val,requires_grad=False))
    print("Packing native expert bank",flush=True)
    with cached_threshold_for_preparation(HERE):
        first=native["_prepare_humming_experts"](raw,torch)
    del raw,weight,scale,val
    # The two arms hold independent device addresses for each captured weight.
    second=copy.copy(first)
    second._parameters={k:torch.nn.Parameter(v.detach().clone(),requires_grad=False) for k,v in first._parameters.items()}
    second._buffers={k:v.clone() for k,v in first._buffers.items()}
    # Keeping one bank on host bounds device use. Build 256 distinct experts by pairing
    # two representative weight banks and assigning per-global-expert BF16 postscales.
    # Cache the immutable native bank on device if sufficient room remains for both graphs.
    # Device reindexing avoids streaming two whole weight sets over PCIe for each logical rank.
    native_fields={name:value.detach() for name,value in list(first.named_parameters())+list(first.named_buffers()) if name.startswith(("w13","w2"))}
    total=sum(value.numel()*value.element_size() for value in native_fields.values())
    on_device=torch.cuda.mem_get_info()[0] > total+1600*1024**2
    originals={name:(value.clone() if on_device else value.cpu()) for name,value in native_fields.items()}
    print("Original expert templates on", "GPU" if on_device else "CPU",flush=True)
    print("Packed bank copies ready",flush=True)
    return {"baseline":first,"candidate":second}, originals


def install(layer, bank, ids):
    gid=torch.from_numpy(ids.astype(np.int64))
    last_indices=getattr(layer,"_bench_installed_indices",{})
    for tag in ("w13","w2"):
        index=gid%32 if tag=="w13" else (gid//32*5+gid%32*7)%32
        for key in ("weight","weight_scale"):
            name=tag+"_"+key
            target=getattr(layer,name)
            source=bank[name]
            if source.is_cuda:
                # The templates are contiguous expert rows. Use hardware D2D copies for the
                # changed slices rather than a tensor-index kernel traversing a GiB bytewise.
                prior=last_indices.get(tag)
                for slot, source_slot in enumerate(index.tolist()):
                    if prior is None or source_slot != prior[slot]:
                        target[slot].copy_(source[source_slot])
            else:
                packed=source.view(torch.uint8).index_select(0,index).view(target.dtype)
                target.copy_(packed)
        last_indices[tag]=index.tolist()
        getattr(layer,tag+"_post_scale").copy_((.125+(gid+(3 if tag=="w2" else 0))*.001).to(torch.bfloat16))
    layer._bench_installed_indices=last_indices
    torch.cuda.synchronize()


def timed(graph):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)*1000


def trial(layers,bank,seed,m,kind,swaps,rounds,drift,rank_count,trace,return_rows):
    heat=demand(seed,kind)
    trace_rows=None
    if trace:
        collection=np.load(HERE/(trace+"_routes.npz"))["arr"]
        trace_rows=collection[seed%3,:,seed%collection.shape[2]].astype(np.int32)
    training=trace_rows[:2048] if trace_rows is not None else draw(seed+41,2048,heat)
    learned,actions=select(route_counts(training),swaps)
    if drift: heat=demand(seed+823,kind)
    placements={"baseline":NATIVE,"candidate":learned}
    x=torch.empty(m,4096,device="cuda",dtype=torch.bfloat16)
    gids=torch.empty(m,8,device="cuda",dtype=torch.int32)
    prob=torch.empty(m,8,device="cuda",dtype=torch.float32)
    norm=torch.linspace(.7,1.2,4096,device="cuda",dtype=torch.bfloat16)
    maps={arm:torch.full((257,),-1,device="cuda",dtype=torch.int32) for arm in layers}
    def inputs(i):
        generator=torch.Generator(device="cuda").manual_seed(seed+i+7927)
        x.normal_(std=.023,generator=generator)
        if trace_rows is None:
            routed=draw(seed+907+i,m,heat)
        else:
            start=2048+(i*m)%(len(trace_rows)-2048-m+1)
            routed=trace_rows[start:start+m]
        gids.copy_(torch.from_numpy(routed))
        prob.uniform_(generator=generator)
        prob.div_(prob.sum(-1,keepdim=True))
    inputs(0)
    result={}
    graphs={}
    for arm,layer in layers.items():
        maps[arm][1:].copy_(torch.from_numpy(local_map(placements[arm],0)))
        def execute():
            ids=maps[arm][gids.to(torch.int32).clamp_min(-1)+1]
            y,local=native["_humming_expert_outputs"](layer,x,ids,torch,global_experts=256,defer_downscale=True)
            out=finish(y,prob,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=layer.w2_post_scale)
            return out,y,local
        execute()
        torch.cuda.synchronize()
        print("Capturing",arm,m,kind,flush=True)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            result[arm]=execute()
        graphs[arm]=g
    arms=["baseline","candidate","control"]
    aggregate={arm:torch.zeros(rounds,m,4096,device="cuda",dtype=torch.float32) for arm in arms}
    raw={arm:torch.zeros(m,8,4096,device="cuda",dtype=torch.bfloat16) for arm in layers}
    samples={arm:np.zeros((rank_count,rounds)) for arm in arms}
    raw_return=torch.empty((rounds,m,8,4096),device="cuda",dtype=torch.bfloat16) if return_rows else None
    for rank in range(rank_count):
        started=time.monotonic()
        print("Installing rank",rank,"for",m,kind,flush=True)
        for arm,layer in layers.items():
            install(layer,bank,placements[arm][rank])
            maps[arm][1:].copy_(torch.from_numpy(local_map(placements[arm],rank)))
            graphs[arm].replay()
            graphs[arm].replay()
        torch.cuda.synchronize()
        for i in range(rounds):
            inputs(i)
            order=arms if (rank+i)%2==0 else arms[::-1]
            for arm in order:
                src="baseline" if arm=="control" else arm
                samples[arm][rank,i]=timed(graphs[src])
                out,inter,local=result[src]
                aggregate[arm][i].add_(out)
                if raw_return is not None and arm=="candidate":
                    active=local>=0
                    raw_return[i][active]=inter[active]
                if i==0 and arm!="control":
                    active=local>=0
                    raw[arm][active]=inter[active]
        print("Completed logical rank",rank,"rows",m,"route pattern",kind,"seconds",round(time.monotonic()-started,2),flush=True)
    ref=aggregate["baseline"]
    other=aggregate["candidate"]
    diff=ref-other
    returned=None
    if raw_return is not None:
        restored=torch.zeros_like(ref)
        physical=np.empty(256,dtype=np.int32)
        physical[learned.reshape(-1)]=np.repeat(np.arange(8),32)
        source=torch.tensor(physical,device="cuda")
        byte_counts=[]
        for i in range(rounds):
            inputs(i)
            dest=gids.long()//32
            host=source[gids.long()]
            moved=dest!=host
            nsend=torch.bincount(host[moved].long(),minlength=8)
            nrecv=torch.bincount(dest[moved].long(),minlength=8)
            byte_counts.append(dict(send=(nsend*8192).tolist(),receive=(nrecv*8192).tolist()))
            for rank in range(8):
                original=maps["baseline"].new_tensor(local_map(NATIVE,rank))[gids.long()]
                original_scale=(.125+(torch.arange(rank*32,(rank+1)*32,device="cuda")+3)*.001).to(torch.bfloat16)
                original_out=finish(raw_return[i],prob,norm_weight=norm,eps=1e-8,weight_first=True,
                      slot_expert_ids=original,expert_output_scale=original_scale)
                restored[i].add_(original_out)
        returned=dict(sm86_original_order_exact=torch.equal(ref,restored),error_values=int((ref!=restored).sum()),
                      payload_by_round=byte_counts,weights_migrated_once_bytes=int(len(actions)*2*28311556),
                      scope="Actual recombination with original scored full-width normalization. Raw values are staged on one GPU; communication and the restoration cost are outside the compute timings.")
    return dict(returned=returned,m=m,pattern=kind,drift=drift,trace=trace,rank_count=rank_count,all_ranks=rank_count==8,swap_actions=actions,rounds=rounds,
        rank_us={arm:values.tolist() for arm,values in samples.items()},
        max_rank_medians_us={arm:float(np.median(values,axis=1).max()) for arm,values in samples.items()},
        per_round_max={arm:values.max(axis=0).tolist() for arm,values in samples.items()},
        shuffled_expert_raw_mismatches=int((raw["baseline"].view(torch.int16)!=raw["candidate"].view(torch.int16)).sum()),
        raw_values=raw["baseline"].numel(),control_bits_equal=torch.equal(ref,aggregate["control"]),
        changed_all=all(not torch.equal(ref[i],ref[i-1]) for i in range(1,rounds)),
        finite=bool(torch.isfinite(ref).all() and torch.isfinite(other).all()),
        max_abs=float(diff.abs().max()),rel_l2=float(diff.norm()/ref.norm().clamp_min(1e-20)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=8)
    parser.add_argument("--rows",type=int,nargs="+",default=[128,768])
    parser.add_argument("--patterns",nargs="+",default=["normal","skewed"])
    parser.add_argument("--drift",action="store_true")
    parser.add_argument("--return-rows",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--rank-count",type=int,choices=range(1,9),default=8)
    parser.add_argument("--trace",choices=("glm53","glm52"))
    args=parser.parse_args()
    if args.return_rows and args.rank_count!=8:parser.error("return-rows needs all eight logical ranks")
    assert hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest()==BASE
    layers,bank=banks(args.seed)
    report=dict(seed=args.seed,baseline=BASE,complete=False,rows={},scope="Eight rank computations serialized on one SM86, no real interconnect or migration; peak rank timing is a distributed compute proxy only.")
    if args.resume and args.out.exists():
        old=json.loads(args.out.read_text())
        assert old["seed"]==args.seed and old["baseline"]==BASE
        report["rows"]=old["rows"]
    for m in args.rows:
        for kind in args.patterns:
            key=f"{m}_{kind}"
            if key in report["rows"]:
                prior=report["rows"][key]
                assert prior["drift"]==args.drift and prior["rounds"]==args.rounds
                assert prior["rank_count"]==args.rank_count and prior["trace"]==args.trace
                continue
            row=trial(layers,bank,args.seed,m,kind,4,args.rounds,args.drift,args.rank_count,args.trace,args.return_rows)
            report["rows"][key]=row
            print(key,row["max_rank_medians_us"],"relative_l2",row["rel_l2"],"raw mismatches",row["shuffled_expert_raw_mismatches"],flush=True)
            args.out.write_text(json.dumps(report,indent=2))
            torch.cuda.empty_cache()
    report["complete"]=True
    args.out.write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    faulthandler.dump_traceback_later(60,repeat=True)
    torch.set_num_threads(2)
    with torch.inference_mode():
        main()