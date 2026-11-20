"""Actual frozen deferred_argmax AST + real Gloo; CPU local-max surrogate only.

Proves collective value equivalence and provenance guards, not NCCL ordering or
the GPU softcap implementation. No performance result is produced.
"""
import ast,json,os,socket,sys,types
from typing import Optional,Tuple
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import deferred_sync as runtime
here=Path(__file__).resolve().parent

def peer(rank,world,port):
 torch.set_num_threads(1)
 dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=world,rank=rank)
 records=[];policy=[];guards=[];calls=[];current=[None];stream=[11,False]
 class Group:
  def __init__(self):
   self.world_size=world;self.rank_in_group=rank;self.ranks=list(range(world))
   self.cpu_group=dist.group.WORLD;self.device_group=dist.group.WORLD;self.pymscclpp_comm=None
  def all_gather_into_tensor(self,out,inp):
   calls.append('all_gather');self._all_gather_into_tensor(out,inp)
  def _all_gather_into_tensor(self,out,inp):
   dist.all_gather_into_tensor(out,inp,group=self.device_group)
  def broadcast(self,values,src=0):
   calls.append('broadcast');dist.broadcast(values,src=self.ranks[src],group=self.device_group)
 class Sync:
  def __init__(self,g):self._tp_group=g;self._sites=frozenset({15})
  def sync(self,*args):
   values=args[-1];self._tp_group.broadcast(values);return values
 def local_max(logits,softcap,valid_width,col_offset):
  values=softcap*torch.tanh(logits[:,:valid_width]/softcap)
  value,index=values.max(-1)
  return torch.stack((value,(index+col_offset).to(value.dtype)),-1)
 tree=ast.parse((here/'dflash_sources/logits_processor.py').read_text())
 cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LogitsProcessor')
 method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='deferred_argmax')
 env=dict(torch=torch,softcap_row_argmax=local_max,get_parallel=lambda:types.SimpleNamespace(tp_group=current[0]))
 exec(compile(ast.Module(body=[method],type_ignores=[]),'frozen-deferred-argmax','exec'),env)
 class LogitsProcessor:
  deferred_argmax=env['deferred_argmax']
  vocab_size=world*19-3;final_logit_softcapping=30.0
 for name,attrs in {
   'sglang.srt.distributed.parallel_state':dict(GroupCoordinator=Group),
   'sglang.srt.layers.logits_processor':dict(LogitsProcessor=LogitsProcessor),
   'sglang.srt.speculative.spec_tp_sync':dict(SpecTpSync=Sync),
   'sglang.srt.runtime_context':dict(get_parallel=env['get_parallel'])}.items():
  module=types.ModuleType(name);module.__dict__.update(attrs);sys.modules[name]=module
 runtime._cuda_descriptor=lambda:(9,0)
 runtime._stream_state=lambda:tuple(stream)
 def setup():
  group=Group();current[0]=group;return group,Sync(group),LogitsProcessor()
 try:
  for case in ('off','on','one_flag_off','one_arch_diff','one_subclass','one_override','one_group_diff','one_rank_order','one_mscclpp','one_group_override','one_sync_override','one_sync_sites'):
   group,sync,lp=setup();os.environ['SGLANG_EXPERIMENTAL_DFLASH_DEFERRED_SYNC']='0' if case=='off' or (case=='one_flag_off' and rank==world-1) else '1'
   runtime._cuda_descriptor=lambda:(8,6) if case=='one_arch_diff' and rank==world-1 else (9,0)
   if rank==world-1:
    if case=='one_subclass':lp=type('Derived',(LogitsProcessor,),{})()
    if case=='one_override':lp.deferred_argmax=lambda x:x
    if case=='one_group_diff':current[0]=Group()
    if case=='one_rank_order':group.ranks=list(reversed(group.ranks))
    if case=='one_mscclpp':group.pymscclpp_comm=object()
    if case=='one_group_override':group.all_gather_into_tensor=lambda *x:None
    if case=='one_sync_override':sync.sync=lambda *x:None
    if case=='one_sync_sites':sync._sites=frozenset()
   candidate=runtime.prepare_deferred_agreement(lp,sync)
   assert (candidate is not None)==(case=='on'),(case,rank)
   policy.append(dict(case=case,enabled=candidate is not None))
  runtime._cuda_descriptor=lambda:(9,0);group,sync,lp=setup()
  candidate=runtime.prepare_deferred_agreement(lp,sync);assert candidate is not None
  generator=torch.Generator().manual_seed(8705123+rank)
  for rows in (1,7,8,32,56,128,256):
   for pattern in ('random','ties','signed_zero','positive_inf','negative_inf','one_nan','many_nan','integer'):
    logits=torch.randn(rows,19,generator=generator)
    if pattern=='ties':logits.fill_(1)
    if pattern=='signed_zero':logits.fill_(-0.0 if rank%2 else 0.0)
    if pattern=='positive_inf':logits[:,3]=float('inf')
    if pattern=='negative_inf':logits.fill_(-float('inf'))
    if pattern=='one_nan' and rank==world-1:logits[:,2]=float('nan')
    if pattern=='many_nan':logits[:,rank%19]=float('nan')
    if pattern=='integer':logits=(logits*4).to(torch.int32).float()
    calls.clear();native=lp.deferred_argmax(logits);sync.sync(native)
    assert calls==['all_gather','broadcast']
    calls.clear();value,proof=candidate.produce(lp,sync,logits)
    assert candidate.consume(proof,value,sync)
    assert calls==['all_gather'] and torch.equal(value,native)
    # Subsequent collective keeps matching order without a replacement barrier.
    state=torch.tensor(rank+1,dtype=torch.int64);dist.all_reduce(state)
    assert int(state)==world*(world+1)//2
    records.append(dict(rows=rows,pattern=pattern,exact=True,broadcasts_removed=1))
  def reject(name,func):
   try:func()
   except RuntimeError:guards.append(name)
   else:raise AssertionError('expected reject: '+name)
  logits=torch.randn(8,19,generator=generator)
  value,proof=candidate.produce(lp,sync,logits);candidate.consume(proof,value,sync)
  reject('double-consume',lambda:candidate.consume(proof,value,sync))
  value,proof=candidate.produce(lp,sync,logits)
  reject('different-result-object',lambda:candidate.consume(proof,value.clone(),sync))
  stream[0]=12;reject('different-stream',lambda:candidate.consume(proof,value,sync));stream[0]=11
  reject('sync-owner-replacement',lambda:candidate.consume(proof,value,Sync(group)))
  candidate.consume(proof,value,sync)
  old_value,old_proof=candidate.produce(lp,sync,logits)
  value,proof=candidate.produce(lp,sync,logits+1)
  reject('stale-generation',lambda:candidate.consume(old_proof,old_value,sync));candidate.consume(proof,value,sync)
  reject('producer-replacement',lambda:candidate.produce(LogitsProcessor(),sync,logits))
  lp.deferred_argmax=lambda x:x
  reject('method-mutation',lambda:candidate.produce(lp,sync,logits));del lp.deferred_argmax
  group.ranks=group.ranks[::-1]
  reject('rank-order-mutation',lambda:candidate.produce(lp,sync,logits));group.ranks=group.ranks[::-1]
  current[0]=Group();reject('parallel-context-mutation',lambda:candidate.produce(lp,sync,logits));current[0]=group
  group.pymscclpp_comm=object();reject('transport-mutation',lambda:candidate.produce(lp,sync,logits));group.pymscclpp_comm=None
  sync.sync=lambda *x:None;reject('sync-method-mutation',lambda:candidate.produce(lp,sync,logits));del sync.sync
  sync._sites=frozenset();reject('sync-site-mutation',lambda:candidate.produce(lp,sync,logits));sync._sites=frozenset({15})
  stream[1]=True;value,proof=candidate.produce(lp,sync,logits);assert proof is None;sync.sync(value);stream[1]=False
  guards.append('outer-capture-retains-broadcast')
  # Dynamic settings use a fresh local variable per worker invocation. Three
  # native non-proven producer paths remain synchronized between proven calls.
  dynamic=[]
  for mode in ('deferred-greedy','ordinary-greedy','sampling-adjusted','grammar','deferred-greedy'):
   calls.clear();proof=None
   if mode=='deferred-greedy':value,proof=candidate.produce(lp,sync,logits)
   else:value=torch.full((8,),rank,dtype=torch.int64)
   if proof is not None:candidate.consume(proof,value,sync)
   else:sync.sync(value)
   assert calls==(['all_gather'] if mode=='deferred-greedy' else ['broadcast'])
   dynamic.append(dict(mode=mode,calls=list(calls)))
  # Execute the actual original/patched acceptance methods, including original
  # eager acceptance and commit helpers. This is still a CPU fixture.
  worker_tree=ast.parse((here/'dflash_sources/dflash_worker_v2.py').read_text())
  patched_tree=ast.parse((here/'deferred_sync_integration/sources/python/sglang/srt/speculative/dflash_worker_v2.py').read_text())
  kernel_tree=ast.parse((here/'sync_sources/dflash.py').read_text())
  accept_env=dict(torch=torch,Optional=Optional,Tuple=Tuple,SIMULATE_ACC_LEN=0,
    SpecTpSyncSite=types.SimpleNamespace(DFLASH_ACCEPT_GREEDY=15),_is_all_greedy=lambda x:True)
  for tree,name in ((worker_tree,'_commit_accept'),(kernel_tree,'compute_dflash_correct_drafts_and_bonus')):
   fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
   exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-accept-helper','exec'),accept_env)
  funcs=[]
  for tree in (worker_tree,patched_tree):
   cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='_accept_block' for f in x.body))
   fn=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='_accept_block')
   exec(compile(ast.Module(body=[fn],type_ignores=[]),'worker-accept','exec'),accept_env);funcs.append(accept_env['_accept_block'])
  worker=types.SimpleNamespace(_selector_sample=None,_use_triton_accept_bonus=False,block_size=8,
    _tp_sync=sync,_deferred_agreement=candidate)
  actual_accept=[]
  for bs in (1,4,7,16,32):
   for break_at in range(8):
    logits=torch.randn(bs*8,19,generator=generator)
    value,proof=candidate.produce(lp,sync,logits)
    proposed=torch.zeros((bs,8),dtype=torch.int64);proposed[:,1:]=value.view(bs,8)[:,:-1]
    if break_at<7:proposed[:,break_at+1]=-1
    kwargs=dict(candidates=proposed,next_token_logits=None,sampling_info=None,draft_input=None,
      prefix_lens=torch.arange(bs,dtype=torch.int64)+100,bs=bs,target_predict=value)
    calls.clear();native=funcs[0](worker,**kwargs);assert calls==['broadcast']
    calls.clear();test=funcs[1](worker,target_agreement=proof,**kwargs);assert not calls
    for a,b in zip(native,test):assert a is b if a is None else torch.equal(a,b)
    assert torch.equal(test[0],torch.full((bs,),break_at,dtype=torch.int32))
    actual_accept.append(dict(bs=bs,accept_len=break_at,exact=True))
  if rank==0:
   tag=os.environ.get('DEFERRED_SYNC_TAG','')
   (here/f'deferred_sync{tag}_gloo_tp{world}.json').write_text(json.dumps(dict(passed=True,scope=__doc__,policy=policy,exact_cases=records,lifecycle_guards=guards,dynamic_settings=dynamic,actual_worker_accept=actual_accept,transport='Gloo CPU; no NCCL timing or GPU ordering claim'),indent=2))
 finally:dist.destroy_process_group()

if __name__=='__main__':
 torch.set_num_threads(1)
 for world in (2,4):
  with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
  mp.start_processes(peer,args=(world,port),nprocs=world,join=True,start_method='fork')
  print('PASS deferred provenance GlooTP',world,flush=True)
