"""Distributed CPU qualification of the scored and experimental row decompositions."""
import argparse
import ast
import datetime
import json
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

here = Path(__file__).resolve().parent

def functions(path, names, namespace):
    syntax = ast.parse((here / path).read_text())
    selected = [node for node in ast.walk(syntax)
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)
    return namespace

class Group:
    ca_comm = None
    def __init__(self, rank, size):
        self.rank_in_group = rank
        self.world_size = size

    def reduce_scatter_tensor(self, out, partials):
        dist.reduce_scatter_tensor(out, partials)

    def all_gather(self, local, dim):
        outputs = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(outputs, local.contiguous())
        return torch.cat(outputs, dim)

class CpuMath:
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    @staticmethod
    def mm(rows, weights, out_dtype):
        assert rows.dtype == weights.dtype == torch.bfloat16 and out_dtype == torch.float32
        return torch.mm(rows.float(), weights.float())

def compare(a, b):
    return [int((a != b).sum()), a.numel(), float((a.float()-b.float()).norm()),
            float(a.float().norm())]

def distributed(rank, world, url, destination):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=url, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=90))
    methods = functions("muse_spark_gemm_ag.py", [
        "prefill_row_projection", "reduce_scatter_prefill_row_projection",
        "_project_and_gather_rows"], {"torch": CpuMath})
    group = Group(rank, world)
    records = []
    for rows, fixture in [(512,"integral"),(520,"random"),(1024,"sparse"),
                          (16384,"random"),(1024,"cancellation")]:
        generator = torch.Generator().manual_seed(638 + rank)
        x = torch.randn(rows, 64, generator=generator).to(torch.bfloat16)
        if fixture == "integral":
            x = torch.randint(-4, 5, x.shape, generator=generator).to(x.dtype)
        if fixture == "sparse":
            x[torch.rand(rows, generator=generator) < 0.38] = 0
        if fixture == "cancellation":
            x *= (256,1,-256,1,256,1,-256,1)[rank]
            x[:, ::2] = (256,1,-256,1,256,1,-256,1)[rank]
        weights = torch.randn(48, 64, generator=torch.Generator().manual_seed(81)).to(x.dtype)
        reference = x.clone()
        dist.all_reduce(reference)
        candidate = torch.empty_like(x[:rows//world])
        group.reduce_scatter_tensor(candidate, x)
        expected = reference.chunk(world)[rank]
        reductions = compare(expected, candidate)
        baseline_output = methods["prefill_row_projection"](group, reference, weights)
        candidate_output = methods["reduce_scatter_prefill_row_projection"](group, x, weights)
        projections = compare(baseline_output, candidate_output)
        if fixture == "integral":
            assert reductions[0] == projections[0] == 0
        local = torch.tensor(reductions, dtype=torch.float64)
        local[2:] **= 2
        dist.all_reduce(local)
        if rank == 0:
            records.append(dict(rows=rows,fixture=fixture,rank_summed_reduction_unequal=int(local[0]),
                reduction_elements=int(local[1]), reduction_relative_l2=float((local[2]/local[3]).sqrt()),
                projection_unequal=projections[0], projection_elements=projections[1],
                projection_relative_l2=projections[2]/projections[3]))
    if rank == 0:
        Path(destination).write_text(json.dumps({"torch":torch.__version__,"backend":"Gloo",
            "ranks":world,"records":records},indent=2)+"\n")
    dist.destroy_process_group()

def dispatch_checks():
    events = []
    class Output:
        def __init__(self, rows=1024, dtype="bf16", contiguous=True):
            self.shape = (rows,4096)
            self.dtype = dtype
            self.contiguous = contiguous
        def is_contiguous(self):
            return self.contiguous
    regular = lambda *args: events.append("allreduce") or args[0]
    signal = lambda name: lambda *args: events.append(name) or args[1]
    forward = SimpleNamespace(sp_active=False)
    parallel = SimpleNamespace(tp_group=Group(0,8),attn_dp_size=1,attn_cp_size=1)
    deterministic = SimpleNamespace(get=lambda: False)
    gate = lambda group,x,w: x.dtype=="bf16" and x.shape[0]>=512 and x.shape[0]%group.world_size==0
    namespace = functions("muse_spark.py",["_reduce_and_unpool","_can_shard_prefill_row_projection"],{
        "torch":torch, "get_parallel":lambda:parallel,"get_forward":lambda:forward,
        "_is_cpu":False,"envs":SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=deterministic),
        "prefill_row_projection_supported":gate,"gemm_ag_supported":lambda *args:False,
        "gemm_ag":signal("gemm_ag"),"prefill_row_projection":signal("project"),
        "reduce_scatter_prefill_row_projection":signal("reduce_scatter"),"bf16_gemm_dispatch":signal("dense"),
        "tensor_model_parallel_all_reduce":regular})
    class Layer:
        tp_size = 8
        moe_tp_size = 1
        experts = SimpleNamespace(use_marlin=True)
        post_expert_proj = SimpleNamespace(weight=1)
        _can_shard_prefill_row_projection = namespace["_can_shard_prefill_row_projection"]
        _can_shard_small_projection = lambda *args: False
        _defers_unpool_gather = lambda *args: False
    def run(label, output, expected, already=False):
        events.clear()
        namespace["_reduce_and_unpool"](layer,output,already_reduced=already)
        assert events==expected, (label, list(events),expected)
        passed.append(label)
    layer=Layer()
    passed=[]
    run("large",Output(16384),["reduce_scatter"])
    run("512",Output(512),["reduce_scatter"])
    run("nondivisible",Output(753),["allreduce","dense"])
    run("small",Output(32),["allreduce","dense"])
    run("dtype",Output(dtype="fp16"),["allreduce","dense"])
    run("noncontiguous",Output(contiguous=False),["allreduce","project"])
    run("already_reduced",Output(),["project"],already=True)
    forward.sp_active=True
    run("sequence_parallel",Output(),["dense"])
    forward.sp_active=False
    for name in ("attn_cp_size","attn_dp_size"):
        setattr(parallel,name,2)
        run(name,Output(),["allreduce","dense"])
        setattr(parallel,name,1)
    layer.moe_tp_size=2
    run("expert_tp",Output(),["project"])
    layer.moe_tp_size=1
    deterministic.get=lambda:True
    run("deterministic",Output(),["allreduce","dense"])
    deterministic.get=lambda:False
    namespace["gemm_ag_supported"]=lambda *args:True
    run("multicast_owner",Output(),["allreduce","gemm_ag"])
    layer.post_expert_proj=None
    run("no_unpool_weight",Output(),["allreduce"])
    return passed

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--ranks",type=int,choices=[2,4,8],required=True)
    args=parser.parse_args()
    print("Dispatch",dispatch_checks(),flush=True)
    os.environ["USE_LIBUV"]="0"
    # FileStore works on Windows wheels built without libuv.
    with tempfile.TemporaryDirectory(dir=here) as temp:
        url=(Path(temp)/"store").as_uri()
        mp.spawn(distributed,args=(args.ranks,url,str(here/f"gloo_tp{args.ranks}.json")),
                 nprocs=args.ranks,join=True)

