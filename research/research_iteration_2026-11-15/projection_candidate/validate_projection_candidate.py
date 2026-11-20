from pathlib import Path
from types import ModuleType, SimpleNamespace
import argparse, ast, contextlib, hashlib, importlib.util, json, logging, statistics, sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path('/mnt/d/OpenKernelChallenge/serving')
OUT = ROOT / 'research_iteration_2026-11-15/projection_candidate'
HELPER = OUT / 'candidate_sources/python/sglang/srt/layers/communication/muse_spark_prefill_projection.py'

def load():
    # Exercise exact exported helper and the repository's exact gather layout.
    # Only the framework's linear dispatcher is bound to F.linear. At these
    # large shapes the Hopper dispatcher falls through to this same operation.
    name = 'sglang.srt.layers.quantization.unquant'
    module = ModuleType(name)
    module.bf16_gemm_dispatch = F.linear
    sys.modules[name] = module
    spec = importlib.util.spec_from_file_location('projection_candidate', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = ROOT / 'sglang/python/sglang/srt/distributed/parallel_state.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GroupCoordinator')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'all_gather')
    namespace = dict(torch=torch, logger=logging.getLogger('test'),
        is_shm_available=lambda *a: False, all_gather_single=dist.all_gather_into_tensor)
    tree = ast.Module(body=ast.parse('from __future__ import annotations').body+[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return helper, namespace['all_gather']

class TestGroup:
    def __init__(self, rank, world, gather):
        self.rank_in_group, self.world_size = rank, world
        self.local_size = world
        self.device_group = dist.group.WORLD
        self.hpu_communicator = None
        self._gather = gather
    def use_symmetric_memory(self, *a, **k): return contextlib.nullcontext()
    def is_allocation_symmetric(self): return False
    def all_gather(self, value, dim): return self._gather(self, value, dim=dim)

def cpu_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+str(rendezvous), rank=rank, world_size=world)
    helper, gather = load()
    group = TestGroup(rank, world, gather)
    results = []
    with torch.inference_mode():
        for seed in (17, 29):
            for m, k, n in ((1,128,256),(513,128,1024),(777,256,512)):
                torch.manual_seed(seed)
                x = torch.randn(m, k).to(torch.bfloat16)
                w = (torch.randn(n, k)/k**.5).to(torch.bfloat16)
                expected = F.linear(x,w)
                actual = helper.prefill_projection(group,x,w)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                assert actual.is_contiguous() and actual.shape == expected.shape
                results.append(dict(seed=seed,m=m,k=k,n=n,bitwise=True))
    dist.barrier()
    if rank == 0:
        (OUT/f'cpu_collective_world{world}.json').write_text(json.dumps(dict(
            world=world, all_ranks_checked=True, cases=results,
            scope='real Gloo collective; exact candidate helper and source-extracted GroupCoordinator.all_gather; F.linear dispatcher binding; not NCCL'),indent=2))
    dist.destroy_process_group()

def graph(call):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): call()
    torch.cuda.current_stream().wait_stream(stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): value=call()
    return g,value

def elapsed(g,reps=10):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps): g.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/reps

def gpu():
    helper,_=load()
    rows=[]
    torch.manual_seed(20261115)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    with torch.inference_mode():
        for k,n in ((8192,4096),(4096,8192)):
            w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
            for m in (511,512,513,1024,4096,16384):
                x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
                full=F.linear(x,w)
                parts=[F.linear(x,wi) for wi in w.chunk(8)]
                combined=torch.cat(parts,dim=1)
                diff=(combined.float()-full.float())
                rel=diff.norm()/full.float().norm()
                maxabs=diff.abs().max().item()
                assert rel.item()<0.003 and maxabs<0.03125,(m,k,n,rel,maxabs)
                # Unsupported GPU must retain the existing serving path.
                assert not helper.prefill_projection_supported(SimpleNamespace(world_size=8),x,w), 'SM86 must fall back'
                del full,parts,combined,diff
                torch.cuda.empty_cache()
                full_g,full_out=graph(lambda:F.linear(x,w))
                local_g,local_out=graph(lambda:F.linear(x,w[:n//8]))
                gathered=torch.randn(8,m,n//8,device='cuda',dtype=torch.bfloat16)
                layout_g,layout_out=graph(lambda:gathered.movedim(0,1).reshape(m,n))
                samples=[]
                for i in range(8):
                    timings={}
                    order=[('full',full_g),('local',local_g),('layout',layout_g)]
                    if i%2: order.reverse()
                    for name,g in order: timings[name]=elapsed(g,3 if m>=4096 else 10)
                    samples.append(timings)
                med={name:statistics.median(v[name] for v in samples) for name in ('full','local','layout')}
                row=dict(m=m,k=k,n=n,maxabs=maxabs,relative_l2=rel.item(),timing_us=med,
                         communication_budget_us=med['full']-med['local']-med['layout'],
                         samples=samples)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
                del full_g,local_g,layout_g,full_out,local_out,layout_out,gathered,x
            del w
    report=dict(gpu=torch.cuda.get_device_name(),rows=rows,
        helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        scope='local projection compute/layout components only; no communication; NOT candidate or H200 speedup',
        reduced_precision_reduction=False)
    (OUT/'gpu_projection_components.json').write_text(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['cpu','gpu'],required=True)
    args=parser.parse_args()
    if args.mode=='gpu': gpu()
    else:
        for world in (2,4,8):
            rendezvous=OUT/f'gloo_rendezvous_{world}'
            assert not rendezvous.exists()
            mp.start_processes(cpu_worker,args=(world,rendezvous),nprocs=world,start_method='fork',join=True)
            print(f'CPU world={world}: all ranks passed',flush=True)
