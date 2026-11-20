"""Compare the pinned stock Marlin and scored Humming routed FFN shard on shared NVFP4 inputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / "sglang"
COMMIT = "297e54fd105e974cdf401e5b4d20253a22c16f90"
SCORED = "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
MOE = "python/sglang/kernels/ops/moe/"
MODEL = "python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"
MAPPER = "python/sglang/srt/layers/moe/token_dispatcher/standard.py"
NATIVE = "python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
SMALL = MOE + "moe_align_small_numel.py"
FINAL = MOE + "muse_spark_post_expert_finalize_triton.py"
RELEASE = ROOT / "release_2026-11-18"
FROZEN = RELEASE / "candidate_sources"


def stock(path):
    return subprocess.check_output(["git", "-C", str(REPO), "show", COMMIT + ":" + path])


def extract(source, names, scope=None):
    parent = ast.parse(source)
    if scope:
        parent = next(node for node in parent.body if isinstance(node, ast.ClassDef) and node.name == scope)
    functions = [fn for fn in parent.body if isinstance(fn, ast.FunctionDef) and fn.name in names]
    assert {fn.name for fn in functions} == set(names)
    program = ast.parse("from __future__ import annotations").body + functions
    return compile(ast.Module(body=program, type_ignores=[]), "<pinned selected functions>", "exec")


def preflight():
    assert hashlib.sha256((RELEASE / "candidate.diff").read_bytes()).hexdigest() == SCORED
    # Compare the arithmetic and routing sources in the local runtime to pinned
    # blobs. File contents matter on a Windows mount, whose mode bits can differ.
    kernel_root = "python/sglang/kernels/"
    common = [MOE+"moe_wna16_marlin.py", MOE+"moe_align.py",
              kernel_root+"ops/activation/activation.py",kernel_root+"jit/csrc/elementwise/activation.cuh",
              kernel_root+"ops/quantization/gptq_marlin_repack.py",
              "python/sglang/srt/layers/quantization/marlin_utils.py",
              "python/sglang/srt/layers/quantization/marlin_utils_fp4.py",
              "python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py",
              *[kernel_root+"jit/csrc/gemm/marlin_moe/"+name
                for name in ("kernel.h","marlin_template.h","moe_wna16_marlin.cuh")]]
    for path in common:
        local = (REPO/path).read_bytes().replace(b"\r\n",b"\n")
        assert local == stock(path).replace(b"\r\n",b"\n"), path
    paths = {p: stock(p) for p in (MODEL,MAPPER,FINAL,SMALL)}
    funcs = ["_marlin_expert_outputs","_forward_marlin"]
    extract(paths[MODEL], funcs, "MuseSpark1xFp4Experts")
    extract((FROZEN / MODEL).read_bytes(), funcs, "MuseSpark1xFp4Experts")
    extract(paths[MAPPER], ["_map_topk_ids_to_local_experts"])
    return {path: hashlib.sha256(data).hexdigest() for path,data in paths.items()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=813)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--rows", nargs="+",type=int,default=[8,32,128])
    args = parser.parse_args()
    hashes = preflight()
    if args.preflight:
        print(json.dumps(hashes, indent=2))
        return
    assert args.out and args.seed and args.rounds >= 4 and args.rounds % 2 == 0
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/"research_2026-11-18_target_moe"))
    import bench_experts as fixture
    for path in (fixture.OLD/"humming_vendor",fixture.OLD/"phase2",ROOT/"optimization_marlin_tiles_2026-11-15"):
        sys.path.insert(0,str(path))
    from kernel_runtime import bootstrap
    bootstrap(REPO)
    import torch
    from fixture_tuning_cache import cached_threshold_for_preparation
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    assert torch.cuda.get_device_capability() == (8,6)
    modules = args.out / "stock_sources"
    modules.mkdir(exist_ok=True)
    stockmodules = {}
    for rel in (SMALL, FINAL):
        dst = modules / Path(rel).name
        dst.write_bytes(stock(rel))
        stockmodules[rel] = fixture.module("pinned_stock_" + dst.stem, dst)
    align = sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"]
    scoredalign = fixture.module("frozen_scored_small_align", FROZEN/SMALL).moe_align_small_numel
    mapmodule = ModuleType("sglang.srt.layers.moe.token_dispatcher.standard")
    mapmodule.torch = torch
    exec(extract(stock(MAPPER),["_map_topk_ids_to_local_experts"]), mapmodule.__dict__)
    sys.modules[mapmodule.__name__] = mapmodule
    native = ModuleType("sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe")
    native.torch = torch
    exec(extract((FROZEN/NATIVE).read_bytes(),["_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs"]),native.__dict__)
    sys.modules[native.__name__] = native
    for name in ("muse_spark_humming_scale", "muse_spark_humming_split"):
        fixture.module("sglang.kernels.ops.moe."+name, FROZEN/(MOE+name+".py"))
    scoredfinish = fixture.module("frozen_scored_finalize", FROZEN/FINAL)
    torch.manual_seed(args.seed)
    h,e = 4096,32
    raw = torch.nn.Module()
    raw.params_dtype = torch.bfloat16
    raw.moe_runner_config = SimpleNamespace(is_gated=True)
    raw.intermediate_size_per_partition = h
    raw.global_num_experts = 256
    raw.num_local_experts = e
    for stage,n in (("w13",2*h),("w2",h)):
        for key,values in (("weight",torch.randint(0,256,(e,n,h//2),device="cuda",dtype=torch.uint8)),
                           ("weight_scale",torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)),
                           ("weight_scale_2",torch.linspace(.125,.5,e,device="cuda",dtype=torch.bfloat16))):
            setattr(raw,stage+"_"+key,torch.nn.Parameter(values,requires_grad=False))
    del values
    with cached_threshold_for_preparation(args.out):
        humming=native._prepare_humming_experts(raw,torch)
    # Both constructors consume the same raw parameters. Humming repacks into its
    # new module; stock replaces raw's parameter handles only after that returns.
    prepare_moe_nvfp4_layer_for_marlin(raw,group_size=16)
    scored = torch.nn.Module()
    scored.use_humming = True
    scored._humming_layer = humming
    scored.global_num_experts = 256
    norm = torch.linspace(.7,1.2,h,device="cuda",dtype=torch.bfloat16)
    mapping = torch.full((256,),-1,device="cuda",dtype=torch.int32)
    mapping[:e] = torch.arange(e,device="cuda",dtype=torch.int32)
    lookup = torch.cat((mapping.new_full((1,),-1),mapping))
    wrapper_names = ["_marlin_expert_outputs","_forward_marlin"]
    for layer,source,finish in ((raw,stock(MODEL),stockmodules[FINAL]),(scored,(FROZEN/MODEL).read_bytes(),scoredfinish)):
        bindings = {"torch":torch,"fused_post_expert_norm_weighted_sum":finish.fused_post_expert_norm_weighted_sum,
                    "_MARLIN_BLOCK_SIZE_M_CANDIDATES":(8,16,32,48,64)}
        exec(extract(source,wrapper_names,"MuseSpark1xFp4Experts"),bindings)
        for name in wrapper_names:
            setattr(layer,name,MethodType(bindings[name],layer))
        layer._expert_map = mapping
        layer._expert_lookup = lookup
        layer._post_expert_norm = norm
        layer._post_expert_norm_weight = norm
        layer._post_expert_norm_eps = 1e-8
        layer._post_expert_norm_weight_first = True
    report = dict(stock_commit=COMMIT,stock_sha256=hashes,scored_patch_sha256=SCORED,
                  device=torch.cuda.get_device_name(),seed=args.seed,rounds=args.rounds,
                  scope="One 32-of-256 local routed expert shard, H=I=4096 BF16, no collective or full-service measurement",
                  precision="FP32 accumulation; BF16 GEMM output and global scale; Humming disables Stream-K. Cross-kernel reduction order may differ.",
                  shapes={},complete=False)
    receipt = args.out/"calibration.json"
    def save():
        receipt.write_text(json.dumps(report,indent=2)+"\n")
    save()
    for rows in args.rows:
        x = torch.empty((rows,h),device="cuda",dtype=torch.bfloat16)
        ids = torch.empty((rows,8),device="cuda",dtype=torch.int32)
        scores = torch.empty((rows,8),device="cuda")
        def update(seed, distribution):
            gen = torch.Generator(device="cuda").manual_seed(seed)
            x.normal_(std=.023,generator=gen)
            keys = torch.rand(rows,256,device="cuda",generator=gen)
            if distribution == "clustered":
                keys[::4,:8] += 2
            ids.copy_(keys.topk(8).indices)
            scores.uniform_(generator=gen)
            scores.div_(scores.sum(-1,keepdim=True))
        graph, output = {}, {}
        update(args.seed+rows,"uniform")
        for label,layer in (("stock",raw),("scored",scored),("stock_copy",raw)):
            align.moe_align_small_numel = scoredalign if label=="scored" else stockmodules[SMALL].moe_align_small_numel
            layer._forward_marlin(x,scores,ids,norm)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                output[label] = layer._forward_marlin(x,scores,ids,norm)
            graph[label] = g
        for label in graph:
            for _ in range(3):
                graph[label].replay()
        for distribution in ("uniform","clustered"):
            data = dict(order=[],times_us={label:[] for label in graph},numerics=[])
            report["shapes"][f"M{rows}_{distribution}"] = data
            for i in range(args.rounds):
                update(args.seed + rows*231 + i, distribution)
                order = ["stock","scored","stock_copy"] if i%2==0 else ["stock_copy","scored","stock"]
                data["order"].append(order)
                for label in order:
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    events[0].record()
                    graph[label].replay()
                    events[1].record()
                    events[1].synchronize()
                    data["times_us"][label].append(events[0].elapsed_time(events[1])*1000)
                a,b = output["stock"].float(),output["scored"].float()
                if not torch.equal(output["stock"], output["stock_copy"]):
                    reference = output["stock"].clone()
                    duplicate = output["stock_copy"].clone()
                    def compare(left, right):
                        different = left.view(torch.int16) != right.view(torch.int16)
                        finite = torch.isfinite(left) & torch.isfinite(right)
                        errors = (left.float() - right.float()).abs().masked_fill(~finite, 0)
                        affected_rows = different.any(-1)
                        return dict(bit_mismatches=int(different.sum()), rows=int(affected_rows.sum()),
                                    nan_left=int(left.isnan().sum()), nan_right=int(right.isnan().sum()),
                                    inf_left=int(left.isinf().sum()), inf_right=int(right.isinf().sum()),
                                    finite_max_abs=float(errors.max()), active_routes_in_affected_rows=
                                    (ids[affected_rows] < e).sum(-1).bincount(minlength=9).tolist())
                    discrepancy = dict(round=i, seed=args.seed + rows*231 + i,
                                       duplicate=compare(reference, duplicate))
                    for label, previous in (("stock", reference), ("stock_copy", duplicate)):
                        graph[label].replay()
                        discrepancy[label + "_repeat"] = compare(previous, output[label])
                    data.setdefault("failed_controls", []).append(discrepancy)
                    print("Stock control disagreement", rows, distribution, discrepancy, flush=True)
                    save()
                    continue
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                delta = (a-b).abs()
                data["numerics"].append(dict(max_abs=delta.max().item(),relative_l2=(delta.norm()/a.norm().clamp_min(1e-20)).item(),
                                            equal_fraction=(a==b).float().mean().item(),active_routes=(ids<e).sum().item()))
                save()
        del graph,output,g,layer
        torch.cuda.empty_cache()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()