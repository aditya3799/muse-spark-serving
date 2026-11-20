"""Pinned decode registry fill plus frozen preplanned re-copy and a replay reader."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import ModuleType,SimpleNamespace
import torch

here=Path(__file__).resolve().parent
repo=here.parents[1]/"sglang"
manifest=json.loads((here/"manifest.json").read_text())
tree=manifest["parent_tree"]
def read(path):
    return subprocess.check_output(["git","-C",str(repo),"show",tree+":python/sglang/"+path]).decode()

def module(name,**kw):
    package=ModuleType(name)
    package.__dict__.update(kw)
    sys.modules[name]=package
    return package

env=SimpleNamespace(SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH=SimpleNamespace(get=lambda:False))
module("sglang.srt.environ",envs=env)
buffers=module("sglang.srt.model_executor.input_buffers",share_input_buffer=None)
bufsrc=ast.parse(read("srt/model_executor/input_buffers.py"))
node=next(n for n in bufsrc.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
          and n.targets[0].id=="INDEX_SEMANTIC_BUFFERS")
exec(ast.unparse(node),buffers.__dict__)
module("sglang.srt.runtime_context",get_parallel=lambda:SimpleNamespace(num_dp_ranks=1))
smallfile=repo/"python/sglang/kernels/ops/memory/small_copy.py"
assert smallfile.read_text(encoding="utf8")==read("kernels/ops/memory/small_copy.py")
smallname="sglang.kernels.ops.memory.small_copy"
spec=importlib.util.spec_from_file_location(smallname,smallfile)
small=importlib.util.module_from_spec(spec)
sys.modules[smallname]=small
spec.loader.exec_module(small)
registry=module("okc_pinned_graph_registry")
exec(read("srt/model_executor/cuda_graph_buffer_registry.py"),registry.__dict__)
runner=read("srt/model_executor/runner/decode_cuda_graph_runner.py")
copy_node=next(n for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.FunctionDef) and n.name=="_copy_replay_inputs")
scope={}
exec("from __future__ import annotations\n"+ast.unparse(copy_node),scope)
recopy=scope["_copy_replay_inputs"]

def case(bs,padded,seed,rounds,loops):
    n=bs*8
    pn=padded*8
    reg=registry.build_decode_registry(device=torch.device("cuda"),max_bs=padded,max_num_token=pn,
          seq_len_fill_value=1,cache_loc_dtype=torch.int64,share_pool=False)
    ids=reg.get_slot("input_ids").buffer
    pos=reg.get_slot("positions").buffer
    slots=reg.get_slot("out_cache_loc").buffer
    lens=reg.get_slot("seq_lens").buffer
    reqix=reg.get_slot("req_pool_indices").buffer
    owner=SimpleNamespace(raw_num_token=n,_prefilled_inputs=None,buffers=SimpleNamespace(input_ids=ids,positions=pos))
    rng=torch.Generator(device="cuda").manual_seed(seed+bs)
    def make():
        lens_cpu=torch.arange(bs,dtype=torch.int64)+random.Random(seed+bs).randrange(1000,3000)
        return SimpleNamespace(input_ids=torch.randint(0,65535,(n,),generator=rng,device="cuda"),
                           positions=torch.randint(1234,85000,(n,),generator=rng,device="cuda"),
                           req_pool_indices=torch.randperm(bs,generator=rng,device="cuda"),
                           out_cache_loc=torch.randint(0,1729980,(n,),generator=rng,device="cuda"),
                           seq_lens=lens_cpu.cuda(),seq_lens_cpu=lens_cpu)
    fixtures=[make() for _ in range(4)]
    kw=dict(raw_bs=bs,padded_bs=padded,raw_num_tokens=n,padded_num_tokens=pn)
    def fill(b):
        reg.fill_from(b,**kw)
    # Every branch consumes static inputs through an actual replay; the integer readout
    # deliberately excludes target attention, draft generation and AX-to-H200 transfer.
    def reader():
        return ids[:n]+pos[:n]*3+slots[:n]*7+lens[:bs].repeat_interleave(8)+reqix[:bs].repeat_interleave(8)
    fill(fixtures[0])
    reader()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured=reader()
    def stage(b,duplicate):
        fill(b)
        if duplicate:
            recopy(owner,b)
        graph.replay()
    for ix in range(16):
        fb=fixtures[ix%4]
        stage(fb,bool(ix%2))
        expected=fb.input_ids+3*fb.positions+7*fb.out_cache_loc+fb.seq_lens.repeat_interleave(8)+fb.req_pool_indices.repeat_interleave(8)
        torch.testing.assert_close(captured,expected,rtol=0,atol=0)
        if padded!=bs:
            assert pos[n:].count_nonzero().item()==slots[n:].count_nonzero().item()==0
    # Eager registry and re-copy are outside capture, as in model-runner replay.
    results={"baseline":[],"candidate":[],"control":[]}
    device_results={k:[] for k in results}
    for k in range(rounds):
        order=["baseline","candidate","control"] if (seed+k)%2 else ["candidate","baseline","control"]
        for arm in order:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t=time.perf_counter_ns()
            start.record()
            for j in range(loops):
                stage(fixtures[(j+k)%4],arm!="candidate")
            end.record()
            end.synchronize()
            results[arm].append((time.perf_counter_ns()-t)/1000/loops)
            device_results[arm].append(start.elapsed_time(end)*1000/loops)
    return dict(bs=bs,padded=padded,parity=True,wall_us=results,event_us=device_results,
                saved_wall_us=statistics.median(a-b for a,b in zip(results["baseline"],results["candidate"])),
                saved_event_us=statistics.median(a-b for a,b in zip(device_results["baseline"],device_results["candidate"])))

if '--source-only' in sys.argv:
    reg=registry.build_decode_registry(device=torch.device('cpu'),max_bs=4,max_num_token=32,seq_len_fill_value=1,cache_loc_dtype=torch.int64,share_pool=False)
    assert reg.get_slot('input_ids').buffer.shape[0]==32
    print('Pinned registry and Triton source import OK')
    raise SystemExit(0)

p=argparse.ArgumentParser()
p.add_argument("--seed",type=int,required=True)
p.add_argument("--rounds",type=int,default=12)
p.add_argument("--loops",type=int,default=20)
args=p.parse_args()
cases=[case(bs,pad,args.seed,args.rounds,args.loops) for bs,pad in [(1,1),(4,4),(7,8),(16,16),(31,32),(32,32)]]
out=dict(seed=args.seed,rounds=args.rounds,loops=args.loops,torch=torch.__version__,device=torch.cuda.get_device_name(),
         parent=manifest["parent_sha256"],candidate=manifest["candidate_sha256"],cases=cases,
         scope="real registry fill, pinned ID/position re-copy, captured integer reader; no model, FA metadata, or full graph-runner Python overhead")
(here/("gpu_stage_seed%d.json"%args.seed)).write_text(json.dumps(out,indent=2))
print([(c["bs"],round(c["saved_wall_us"],2),round(c["saved_event_us"],2)) for c in cases],flush=True)
