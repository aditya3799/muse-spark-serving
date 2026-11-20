"""Real CPU/Gloo collective validation of isolated selector packing, not NCCL timing."""
from __future__ import annotations
import ast, contextlib, hashlib, json, pathlib, tempfile, types, typing
from probe_selector_pack import ROOT, OUT, SOURCE, get_sources

def worker(rank, world_size, store_path):
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store_path,rank=rank,world_size=world_size)
    group_source=ROOT/'sglang/python/sglang/srt/distributed/parallel_state.py'
    module=ast.parse(group_source.read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='GroupCoordinator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='all_gather')
    code=ast.get_source_segment(group_source.read_text(),method)
    ns={'torch':torch,'Optional':typing.Optional,'List':typing.List,'is_shm_available':lambda *a:False,'all_gather_single':lambda out,inp,group:dist.all_gather_into_tensor(out,inp,group=group)}
    exec(compile(code,str(group_source),'exec'),ns)
    fake=types.SimpleNamespace(world_size=world_size,hpu_communicator=None,local_size=world_size,device_group=dist.group.WORLD,use_symmetric_memory=lambda *a,**kw:contextlib.nullcontext(),is_allocation_symmetric=lambda:False)
    gather=types.MethodType(ns['all_gather'],fake)
    rows=[]
    for n in (1,8,32,256):
      for k in (1,16):
       for dtype in (torch.bfloat16,torch.float16,torch.float32):
        for pattern in ('random','ties','zeros'):
         for large in (False,True):
          vocab=(1<<24)+8 if large else 202048
          generator=torch.Generator().manual_seed(316+rank+n+k)
          logits=torch.randn(n,257,generator=generator).to(dtype)
          if pattern=='ties':logits=logits.round()
          if pattern=='zeros':logits.zero_();logits[:,::2]=-0.0
          head=types.SimpleNamespace(weight=torch.empty(1),quant_method=None,org_vocab_size=vocab,shard_indices=types.SimpleNamespace(num_org_elements=257,org_vocab_start_index=rank*(vocab//world_size)))
          outputs=[];counts=[]
          for code in get_sources():
           calls=[]
           def collect(x,dim=-1):calls.append((tuple(x.shape),str(x.dtype)));return gather(x,dim)
           scope={'torch':torch,'nn':torch.nn,'Tuple':typing.Tuple,'Optional':typing.Optional,'get_parallel':lambda:types.SimpleNamespace(tp_size=world_size),'should_apply_lm_head_quant_method':lambda *a:False,'is_dense_head_weight':lambda x:True,'_project_candidate_logits':lambda *a,**kw:logits,'_radix_topk':lambda x,k:torch.topk(x,k,dim=-1),'tensor_model_parallel_all_gather':collect}
           exec(compile(code,str(SOURCE),'exec'),scope)
           outputs.append(scope['candidate_topk'](torch.empty(n,1),head,k));counts.append(len(calls))
          for a,b in zip(outputs[0][:2],outputs[1][:2]):assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
          assert counts==[2,2 if large else 1]
          rows.append({'rows':n,'k':k,'dtype':str(dtype),'pattern':pattern,'large_vocab_fallback':large,'bitwise':True,'collectives':counts})
    dist.barrier()
    if rank==0:
      report={'status':'REAL_GLOO_COLLECTIVE_CORRECTNESS_NOT_CUDA_OR_PERFORMANCE','ranks':world_size,'cases_per_rank':len(rows),'all_bitwise':True,'group_source_sha256':hashlib.sha256(group_source.read_bytes()).hexdigest(),'selector_source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'rows':rows,'limits':['CPU/Gloo only; actual pinned all_gather method executed with CPU shared-memory fastpath disabled','No NCCL, symmetric-memory allocator, graph capture or multi-GPU/H200 evidence','Fewer collectives does not imply measured serving gain']}
      (OUT/'SELECTOR_PACK_GLOO.json').write_text(json.dumps(report,indent=2)+'\n')
      print(json.dumps({k:report[k] for k in ('status','ranks','cases_per_rank','all_bitwise')}),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='okc-selector-gloo-') as folder:
        mp.spawn(worker,args=(2,str(pathlib.Path(folder)/'store')),nprocs=2,join=True)
