from pathlib import Path
from types import SimpleNamespace
import argparse, ast, contextlib, hashlib, json, logging, statistics
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path('/mnt/d/OpenKernelChallenge/serving')
OUT=ROOT/'swarm_2026-11-16/projection_communication'
HELPER=OUT/'candidate_sources/python/sglang/srt/layers/communication/muse_spark_gemm_ag.py'

def extract(path,names,namespace,class_name=None):
    tree=ast.parse(path.read_text())
    body=tree.body
    if class_name:
        body=next(x for x in body if isinstance(x,ast.ClassDef) and x.name==class_name).body
    nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    unit=ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit),str(path),'exec'),namespace)
    return namespace

def helper(cpu=False):
    ns={'torch':torch}
    if cpu:
        # CPU lacks CUDA's out_dtype=FP32; model the same complete FP32 reduction.
        ns['torch']=SimpleNamespace(mm=lambda a,b,out_dtype:torch.mm(a.float(),b.float()),float32=torch.float32)
    return extract(HELPER,{'prefill_row_projection'},ns)['prefill_row_projection']

class GlooGroup:
    def __init__(self,rank,world,gather):
        self.rank_in_group,self.world_size,self.local_size=rank,world,world
        self.device_group=dist.group.WORLD
        self.hpu_communicator=None
        self._gather=gather
    def use_symmetric_memory(self,*a,**kw): return contextlib.nullcontext()
    def is_allocation_symmetric(self): return False
    def all_gather(self,x,dim): return self._gather(self,x,dim)

def worker(rank,world,rv):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(rv),rank=rank,world_size=world)
    ns=dict(torch=torch,logger=logging.getLogger('test'),is_shm_available=lambda *a:False,all_gather_single=dist.all_gather_into_tensor)
    gather=extract(ROOT/'sglang/python/sglang/srt/distributed/parallel_state.py',{'all_gather'},ns,'GroupCoordinator')['all_gather']
    group=GlooGroup(rank,world,gather)
    run=helper(cpu=True)
    rows=[]
    with torch.inference_mode():
        for seed in (7,19):
            for m,k,n,noncontig in ((world*2,128,256,False),(512,256,128,True),(1024,128,256,False)):
                torch.manual_seed(seed)
                x=torch.randn(m,k*(2 if noncontig else 1),dtype=torch.bfloat16)
                if noncontig:x=x[:,::2]
                w=(torch.randn(n,k)/k**.5).to(torch.bfloat16)
                output=run(group,x,w)
                expected=torch.cat([torch.mm(xi.float(),w.float().t()).to(x.dtype) for xi in x.chunk(world)],dim=0)
                torch.testing.assert_close(output,expected,atol=0,rtol=0)
                baseline=F.linear(x,w)
                rel=((output.float()-baseline.float()).norm()/baseline.float().norm()).item()
                maxabs=(output.float()-baseline.float()).abs().max().item()
                assert rel <= 0.003 and maxabs <= 0.03125
                assert output.is_contiguous()
                rows.append(dict(seed=seed,m=m,k=k,n=n,noncontiguous=noncontig,gather_bitwise=True,baseline_relative_l2=rel,baseline_maxabs=maxabs))
    dist.barrier()
    if rank==0:(OUT/f'cpu_world{world}.json').write_text(json.dumps(dict(world=world,all_ranks_checked=True,rows=rows,scope='Real Gloo/source-extracted SGLang gather; CPU FP32 mm substitute for CUDA out_dtype, not NCCL or GPU arithmetic.'),indent=2))
    dist.destroy_process_group()

def graph(call):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):call()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):value=call()
    return graph,value

def elapsed(g,reps=3):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):g.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/reps

class LocalGroup:
    def __init__(self,rank,world):self.rank_in_group,self.world_size=rank,world
    def all_gather(self,x,dim):
        assert dim==0
        return x

def gpu():
    run=helper();rows=[]
    with torch.inference_mode():
      for k,n in ((8192,4096),(4096,8192)):
       torch.manual_seed(96)
       w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
       for m in (512,1024,4096,16384):
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        # Serial local rank computations exercise exact helper without claiming NCCL.
        g,combined=graph(lambda:torch.cat([run(LocalGroup(r,8),x,w) for r in range(8)],dim=0))
        errors=[]
        for reduction in (False,True):
         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduction
         for seed in (11,29,43):
          torch.manual_seed(seed);x.copy_(torch.randn_like(x));g.replay()
          baseline=F.linear(x,w)
          reference=F.linear(x.float(),w.float())
          diff=combined.float()-baseline.float()
          rel=(diff.norm()/baseline.float().norm()).item()
          maxabs=diff.abs().max().item()
          cand_ref=((combined.float()-reference).norm()/reference.norm()).item()
          base_ref=((baseline.float()-reference).norm()/reference.norm()).item()
          passed=rel<=0.003 and maxabs<=0.03125 and cand_ref<=base_ref+1e-5
          errors.append(dict(seed=seed,reduction=reduction,relative_l2=rel,max_abs=maxabs,candidate_ref_l2=cand_ref,baseline_ref_l2=base_ref,passed=passed))
          del baseline,reference,diff
        del g,combined
        torch.cuda.empty_cache()
        gathered=torch.empty(8,m,n//8,device='cuda',dtype=torch.bfloat16)
        calls={'full':lambda:F.linear(x,w),'full_control':lambda:F.linear(x,w),'row_fp32':lambda:run(LocalGroup(0,8),x,w),'column_fp32':lambda:torch.mm(x,w[:n//8].t(),out_dtype=torch.float32).to(x.dtype),'column_layout':lambda:gathered.movedim(0,1).reshape(m,n)}
        gs={};values=[]
        for name,call in calls.items(): gg,v=graph(call);gs[name]=gg;values.append(v)
        samples=[]
        for i in range(6):
            order=list(gs)
            if i%2:order.reverse()
            samples.append({name:elapsed(gs[name]) for name in order})
        med={name:statistics.median(s[name] for s in samples) for name in gs}
        row=dict(m=m,k=k,n=n,errors=errors,median_us=med,samples=samples,row_communication_budget_us=med['full']-med['row_fp32'],column_communication_budget_us=med['full']-med['column_fp32']-med['column_layout'])
        rows.append(row)
        print(json.dumps(dict(m=m,k=k,n=n,passed=all(e['passed'] for e in errors),median_us=med)),flush=True)
        del gs,values,gg,v,x,gathered,calls
       del w
    report=dict(rows=rows,gpu=torch.cuda.get_device_name(),helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),scope='Single RTX3050 serial rank compute and layout only; no network, no H200 performance evidence.',passed=all(e['passed'] for r in rows for e in r['errors']))
    (OUT/'gpu_validation_and_timings.json').write_text(json.dumps(report,indent=2))
    assert report['passed'],'numerical screen failed'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['cpu','gpu'],required=True);args=parser.parse_args()
    if args.mode=='gpu':gpu()
    else:
      for world in (2,4,8):
        rv=OUT/f'gloo_v2_{world}'
        assert not rv.exists()
        mp.start_processes(worker,args=(world,rv),nprocs=world,start_method='fork',join=True)
        print(f'Gloo world {world}: passed',flush=True)
