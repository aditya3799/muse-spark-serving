"""Check the candidate's actual projection, weight loading, and feature ordering."""
import logging
import argparse
import ast
from contextlib import nullcontext
from pathlib import Path
import json
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["distributed", "gpu"])
parser.add_argument("candidate", type=Path)
parser.add_argument("repo", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
rank, world = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))

def extract(path, node_name, method=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == node_name)
    if method is not None:
        node = next(n for n in node.body if getattr(n, "name", None) == method)
    return ast.unparse(node)

srt = args.repo / "python/sglang/srt"
def default_weight_loader(param, value):
    param.data.copy_(value)

def set_weight_attrs(p, attrs):
    for key, value in attrs.items():
        setattr(p, key, value)

# Load the pinned narrowing/copy logic with rank resolution injected for this fixture.
loader_src = extract(srt/"model_loader/weight_utils.py", "sharded_weight_loader")
loader_src = loader_src.replace("from sglang.srt.layers.linear import resolve_linear_parallel_group",
                                "resolve_linear_parallel_group = lambda name: (rank, world)")
is_cpu = lambda: True
exec("from __future__ import annotations\n" + loader_src)
exec("from __future__ import annotations\n" + extract(args.candidate, "_ShardedContextProjection"))
exec("from __future__ import annotations\n" + extract(args.candidate,"DFlashDraftModel","load_weights"))
exec("from __future__ import annotations\n" + extract(args.candidate,"DFlashDraftModel","project_target_hidden"))

class ProjectionFixture(nn.Module):
    load_weights = load_weights
    project_target_hidden = project_target_hidden
    def __init__(self, k, n, group):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=n)
        self.num_context_features = k // n
        self.fc = _ShardedContextProjection(k, n, group)
        self.hidden_norm = nn.RMSNorm(n, eps=1e-6)
        self.quantizable = False
        self.is_nemotron_35_draft = False
        self.projector_type = "linear"

def save(data):
    if rank == 0:
        args.out.write_text(json.dumps(data,indent=2)+"\n")
        print(json.dumps(data), flush=True)

def distributed():
    dist.init_process_group("gloo")
    from torch.distributed import all_gather_single as gloo_all_gather
    def all_gather_single(out, data, group):
        gloo_all_gather(out,data,group=group)
    class Group:
        rank_in_group = rank
        world_size = world
        local_size = world
        device_group = dist.group.WORLD
        hpu_communicator = None
        use_symmetric_memory = staticmethod(lambda *a, **kw: nullcontext())
        is_allocation_symmetric = staticmethod(lambda: False)
        @staticmethod
        def all_gather_into_tensor(out, data):
            # Gloo transports CPU data between independent ranks on this one-GPU host.
            buf = torch.empty(out.shape, dtype=out.dtype)
            gloo_all_gather(buf, data.cpu(), group=dist.group.WORLD)
            out.copy_(buf)
    env = dict(globals(),all_gather_single=all_gather_single,is_shm_available=lambda *a:False)
    exec("from __future__ import annotations\n" + extract(srt/"distributed/parallel_state.py","GroupCoordinator","all_gather"), env)
    Group.all_gather = env["all_gather"]
    fc = ProjectionFixture(512,128,Group()).to(device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(354)
    checks = 0
    for alias in ["encoder.fc.weight","fc.weight","model.fc.weight"]:
        w = torch.randint(-3,4,(128,512)).bfloat16()
        fc.load_weights([(alias,w)])
        assert torch.equal(fc.fc.weight, w.chunk(world)[rank].cuda())
        checks += 1
        for m in (1,8,13,64):
            x = torch.randint(-2,3,(m,512)).bfloat16()
            x = x.cuda()
            ref = fc.hidden_norm(F.linear(x,w.cuda()))
            actual = fc.project_target_hidden(x)
            assert torch.equal(actual,ref)
            checks += 1
    bad = w.chunk(world)[rank]
    try:
        fc.load_weights([("encoder.fc.weight",bad)])
    except ValueError as e:
        assert "fc.weight shape mismatch" in str(e)
        checks += 1
    else:
        raise AssertionError("truncated checkpoint passed")
    dist.barrier()
    save(dict(mode="two CUDA processes on one GPU with CPU/Gloo collective transport",ranks=world, checks_per_rank=checks,
              note="Extracted candidate and pinned SGLang methods. Runtime group lookup and allocator are fixtures."))
    dist.destroy_process_group()

def gpu():
    torch.manual_seed(423)
    torch.cuda.manual_seed_all(423)
    device = "cuda"
    k,n = 40960,8192
    baseline = torch.randn((n,k),device=device,dtype=torch.bfloat16) * (k ** -0.5)
    timings = []
    for m in (8,32,128,256,753,1024):
        x = torch.randn((m,k),device=device,dtype=torch.bfloat16)
        def whole():
            return F.linear(x,baseline)
        def columns(w):
            return torch.mm(x,w.t(),out_dtype=torch.float32).to(x.dtype)
        def shard():
            return columns(baseline[:n//8])
        full = whole()
        gathered = torch.cat([columns(ws) for ws in baseline.chunk(8)],dim=-1)
        assert torch.isfinite(full).all() and torch.isfinite(gathered).all()
        error = (full.float()-gathered.float()).abs()
        rel = error.norm().item()/full.float().norm().item()
        sampled = torch.arange(7,n,191,device=device)
        oracle = (x.double() @ baseline[sampled].double().T).bfloat16().float()
        sample_error = (gathered[:,sampled].float()-oracle).norm().item()/oracle.norm().item()
        stock_error = (full[:,sampled].float()-oracle).norm().item()/oracle.norm().item()
        if sample_error > 0.001:
            raise AssertionError((m,sample_error,stock_error))
        # Use alternating paired event timings, include independent full/full controls.
        for _ in range(3):
            whole(); shard()
        def timed(fn):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)/3
        pairs = []
        controls = []
        for turn in range(9):
            if turn % 2:
                sh,base = timed(shard),timed(whole)
            else:
                base,sh = timed(whole),timed(shard)
            pairs.append([base,sh])
            controls.append(timed(whole)/base)
        timings.append(dict(m=m, bitwise_fraction=(error==0).float().mean().item(),
                       max_abs=error.max().item(), relative_l2=rel, stock_parity_gate=rel<=0.004,
                       sampled_fp64_rel_l2=sample_error,stock_sampled_fp64_rel_l2=stock_error,
                       full_ms=statistics.median(p[0] for p in pairs),
                       shard_ms=statistics.median(p[1] for p in pairs),
                       median_paired_ratio=statistics.median(b/s for b,s in pairs),
                       aa_median=statistics.median(controls),pairs_ms=pairs, aa_ratios=controls))
        print("Completed",m,flush=True)
    save(dict(mode="single GPU projection only, gather excluded",device=torch.cuda.get_device_name(),
              torch_version=torch.__version__,weight_shape=[n,k],timings=timings))

with torch.inference_mode():
    if args.mode=="gpu":
        gpu()
    else:
        distributed()

