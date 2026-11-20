"""Compare two expert GEMMs against the scored light/heavy split on the same inputs."""
import argparse
import ast
import hashlib
import importlib.util
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "release_2026-11-18"
SOURCE = RELEASE / "candidate_sources/python/sglang"
OLD = ROOT / "swarm_2026-11-16/moe"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def paired_ci(pairs):
    rng = random.Random(813)
    ratios = [a/b for a, b in pairs]
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return [draws[49], statistics.median(ratios), draws[1949]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=[8,32,128,256,753])
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    for path in (OLD/"humming_vendor", OLD/"phase2", ROOT/"optimization_marlin_tiles_2026-11-15"):
        sys.path.insert(0,str(path))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    import torch
    from fixture_tuning_cache import cached_threshold_for_preparation
    moe = SOURCE/"kernels/ops/moe"
    module("sglang.kernels.ops.moe.muse_spark_humming_scale",moe/"muse_spark_humming_scale.py")
    split = module("sglang.kernels.ops.moe.muse_spark_humming_split",moe/"muse_spark_humming_split.py")
    finish = module("scored_finalize",moe/"muse_spark_post_expert_finalize_triton.py").fused_post_expert_norm_weighted_sum
    path = SOURCE/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
    tree = ast.parse(path.read_text())
    names = ("_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs")
    funcs = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    assert set(funcs)==set(names)
    ns = {"torch":torch}
    exec(compile(ast.Module(body=list(funcs.values()),type_ignores=[]),str(path),"exec"),ns)
    # Change only config selection in the unpartitioned adapter. The native math,
    # alignment and allocation still receive every route and its full logical shape.
    original = ast.get_source_segment(path.read_text(),funcs["_humming_expert_outputs_default"])
    selection = "selected = {s: next((c for lo, hi, c in cfg if lo < valid <= hi)) for s, cfg in configs.items()}"
    assert original.count(selection)==1
    modified = original.replace("_humming_expert_outputs_default(", "single_pass(").replace("*, defer_downscale=False", "*, selected, defer_downscale=False").replace(selection,"# Selected by the caller.")
    exec(compile(modified,str(path)+":single_pass","exec"),ns)
    digest = hashlib.sha256((RELEASE/"candidate.diff").read_bytes()).hexdigest()
    assert digest=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
    report = dict(device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                  patch=digest,hidden=4096,intermediate=4096,local_experts=32,top_k=8,
                  prediction="A single local-work-tuned W13/W2 pair should reduce idle heavy launches on small or balanced batches; concentrated routing may regress.",
                  rows=[],status="running")
    def save():
        (args.out/"results.json").write_text(json.dumps(report,indent=2))
    save()
    torch.manual_seed(127)
    raw = torch.nn.Module()
    h,e = 4096,32
    for stage,n in (("w13",2*h),("w2",h)):
        for key,t in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                      ("weight_scale",torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                      ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
            setattr(raw,stage+"_"+key,torch.nn.Parameter(t,requires_grad=False))
    with cached_threshold_for_preparation(args.out):
        layer = ns["_prepare_humming_experts"](raw,torch)
    del raw,t
    norm = torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
    for m in args.rows:
        plan = split._choose_large_prefill_configs(layer,m*8,256)
        if plan is None:
            report["rows"].append(dict(m=m,status="scored path already unpartitioned"))
            save()
            continue
        x = torch.empty((m,h),device="cuda",dtype=torch.bfloat16)
        ids = torch.empty((m,8),device="cuda",dtype=torch.int32)
        scores = torch.empty((m,8),device="cuda")
        def inputs(seed,pattern):
            gen = torch.Generator().manual_seed(seed)
            keys = torch.rand((m,256),generator=gen)
            if pattern=="cluster":
                keys[::2,:8]+=2
            routes = keys.topk(8).indices
            ids.copy_(torch.where(routes<32,routes,-1))
            torch.manual_seed(seed)
            x.normal_(std=.023)
            scores.copy_(torch.rand_like(scores))
            scores.div_(scores.sum(-1,keepdim=True))
        def run(single):
            if single:
                down,local = ns["single_pass"](layer,x,ids,torch,selected=plan["light"],defer_downscale=True)
            else:
                down,local = ns["_humming_expert_outputs"](layer,x,ids,torch,defer_downscale=True,global_experts=256)
            return finish(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local,expert_output_scale=layer.w2_post_scale)
        inputs(37,"uniform")
        graphs,outputs=[],[]
        for single in (False,True):
            run(single)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output=run(single)
            graphs.append(graph)
            outputs.append(output)
        def timed(graph):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)*1000
        for pattern in ("uniform","cluster"):
            verdict=[]
            for seed in (43,73,137):
                inputs(seed,pattern)
                for graph in graphs:
                    graph.replay()
                assert torch.isfinite(outputs[0]).all() and torch.isfinite(outputs[1]).all()
                verdict.append(torch.equal(*outputs))
            reference=outputs[0].clone()
            # Falsify a stale-output implementation on a new activation.
            x.neg_()
            for graph in graphs:
                graph.replay()
            assert not torch.equal(outputs[0],reference)
            verdict.append(torch.equal(*outputs))
            # Keep correctness failures in the ledger even when a kernel is fast.
            pairs,controls=[],[]
            for graph in graphs:
                for _ in range(3):
                    graph.replay()
            choose=random.Random(91)
            for i in range(20):
                inputs(900+i,pattern)
                order=[0,1]
                choose.shuffle(order)
                times={}
                for index in order:
                    times[index]=timed(graphs[index])
                control=timed(graphs[0])
                pairs.append([times[0],times[1]])
                controls.append(control/times[0])
            row=dict(m=m,pattern=pattern,cutoff=plan["cutoff"],exact_checks=verdict,median_baseline_us=statistics.median(a for a,b in pairs),median_single_us=statistics.median(b for a,b in pairs),ratio_ci=paired_ci(pairs),control_median=statistics.median(controls),pairs_us=pairs)
            report["rows"].append(row)
            print(m,pattern,row["ratio_ci"],verdict,flush=True)
            save()
        del graphs,outputs,graph,output,reference
        torch.cuda.empty_cache()
    report["status"]="complete"
    save()

if __name__ == "__main__":
    main()

