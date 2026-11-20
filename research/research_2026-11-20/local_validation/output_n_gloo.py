"""TP8 real Gloo control/fallback test; tiny exact CPU arithmetic, mocked vendor plans."""
import json,tempfile,uuid,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys,types,importlib
SOURCE=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_final/assembled/python/sglang')
for name,path in [('sglang',SOURCE),('sglang.srt',SOURCE/'srt'),('sglang.srt.speculative',SOURCE/'srt/speculative')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
adapter=importlib.import_module('sglang.srt.speculative.dflash_projection_n.adapter')
HERE=Path(__file__).resolve().parent

def worker(rank,world,path):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method='file://'+path,world_size=world,rank=rank)
 adapter.compatible=lambda x,w,world:True
 adapter.runtime_state=lambda:(('cublas',32),True,False,False)
 torch.cuda.current_stream=lambda:SimpleNamespace(cuda_stream=1)
 torch.cuda.is_current_stream_capturing=lambda:False
 torch.cuda.synchronize=lambda *args:None
 class Group:
  world_size=world;rank_in_group=rank;cpu_group=dist.group.WORLD
  def __init__(self):self.calls=0
  def all_gather(self,x,dim):
   self.calls+=1;parts=[torch.empty_like(x) for _ in range(world)];dist.all_gather(parts,x)
   return torch.cat(parts,dim=dim)
 gen=torch.Generator().manual_seed(418793)
 model=torch.nn.Linear(16,16,bias=False,dtype=torch.bfloat16)
 with torch.no_grad():model.weight.copy_(torch.randint(-2,3,model.weight.shape,generator=gen)*.125)
 x=(torch.randint(-2,3,(8,16),generator=gen)*.125).bfloat16();reference=model(x)
 class Plan:
  def __init__(self,reject=False,wrong=False):
   self.reject=reject;self.wrong=wrong;self.m=8;self.k=16;self.n=16;self.world=8;self.index=0;self.device=x.device
   self.workspace=torch.empty(32,dtype=torch.uint8)
  def invoke(self,index,full,w,value,out,scratch):
   if self.reject:return 15
   out.copy_(torch.nn.functional.linear(value,w))
   if self.wrong:out[0,0]+=1
   return 0
  def close(self):pass
 receipts={}
 for scenario in ('ordinary','one_rank_declines','one_rank_discovery_exception','one_rank_vendor_failure','strict_bad_rank','one_rank_empty','inference_weight'):
  saved_weight=model.weight
  if scenario=='inference_weight':
   with torch.inference_mode():model.weight=torch.nn.Parameter(saved_weight.clone(),requires_grad=False)
  group=Group();router=adapter.OutputNRouter(model,group,mode='shadow' if scenario=='strict_bad_rank' else 'qualified')
  assert router.prepare_registration([x.shape])[0]
  def discover(examples,size):
   if scenario=='one_rank_declines' and rank==7:return None,{}
   if scenario=='one_rank_discovery_exception' and rank==7:raise ValueError('injected discovery error')
   return Plan(reject=scenario=='one_rank_vendor_failure' and rank==7,wrong=scenario=='strict_bad_rank' and rank==0),{}
  try:
   metadata=router.register([] if scenario=='one_rank_empty' and rank==7 else [x,x.neg()],discover_fn=discover)
  except RuntimeError as error:
   assert scenario=='one_rank_discovery_exception' and 'qualification aborted' in str(error)
   assert group.calls==0
   receipts[scenario]=dict(coherent_fatal_abort=True,gather_calls=0)
   router.retire(graphs_destroyed=True);model.weight=saved_weight;continue
  assert scenario!='one_rank_discovery_exception', 'fatal error silently swallowed'
  with torch.no_grad():out=router(x)
  assert torch.equal(out,reference),(scenario,rank)
  expected=0 if scenario in ('one_rank_declines','one_rank_discovery_exception','one_rank_empty') else 1
  assert group.calls==expected
  for m in (1,4):
   before=group.calls;assert torch.equal(router(x[:m]),model(x[:m]));assert group.calls==before
  receipts[scenario]=dict(exact=True,gather_calls=group.calls,globally_enabled=metadata.get('globally_enabled',False))
  router.retire(graphs_destroyed=True)
  model.weight=saved_weight
 # Rank-asymmetric pointer rejection and capture on an unwarmed stream both
 # still participate in the same output-column gather.
 for scenario in ('one_rank_pointer_guard','one_rank_unwarmed_capture_stream'):
  group=Group();router=adapter.OutputNRouter(model,group,mode='qualified');router.prepare_registration([x.shape])
  router.register([x,x.neg()],discover_fn=lambda *args:(Plan(),{}))
  if rank==7:
   if scenario.endswith('pointer_guard'):router.entries[tuple(x.shape)].weight_pointer+=256
   else:
    torch.cuda.current_stream=lambda:SimpleNamespace(cuda_stream=99)
    torch.cuda.is_current_stream_capturing=lambda:True
  with torch.no_grad():out=router(x)
  assert torch.equal(out,reference) and group.calls==1
  receipts[scenario]=dict(exact=True,gather_calls=1)
  torch.cuda.current_stream=lambda:SimpleNamespace(cuda_stream=1)
  torch.cuda.is_current_stream_capturing=lambda:False
  router.retire(graphs_destroyed=True)
 group=Group();router=adapter.OutputNRouter(model,group,mode='qualified')
 assert not router.prepare_registration([x.shape] if rank==0 else [(16,16)])[0]
 metadata=router.register([x,x.neg()]);assert not metadata['globally_enabled']
 receipts['mismatched_startup_manifest']=dict(coherent_rejection=True)
 if rank==0:
  (HERE/'output_n_actual_gloo.json').write_text(json.dumps(dict(correct=True,world=8,scope=__doc__,cases=receipts,
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 dist.destroy_process_group()

if __name__=='__main__':
 torch.set_num_threads(1)
 path=str(Path(tempfile.gettempdir())/('okc-output-n-'+uuid.uuid4().hex))
 mp.start_processes(worker,args=(8,path),nprocs=8,join=True,start_method='fork')
 print('Output-N TP8 Gloo controls passed',flush=True)
