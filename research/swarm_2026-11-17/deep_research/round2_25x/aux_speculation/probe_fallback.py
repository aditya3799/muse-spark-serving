import argparse,importlib.util,json,sys,types,os,hashlib
from pathlib import Path
import torch
from interleaved_dispatch import activate
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--targeted',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
r={'status':'RUNNING','checks':[],'scope':'Native original scale_active_rows plus SiLU versus same scaled interleaved values with load-only activation adapter. No full W13 GEMM/finalizer or serving integration.'}
def save():a.out.write_text(json.dumps(r,indent=2)+'\n')
def phase(name):r['phase']=name;save();print(name,flush=True)
def check(name,ok):r['checks'].append({'name':name,'passed':bool(ok)});save();assert ok,name
x=torch.zeros(3,32,dtype=torch.bfloat16);out=torch.empty(3,16,dtype=torch.bfloat16);ids=torch.zeros(3,dtype=torch.int32);calls=[]
for layout in ('canonical','gate_up_interleaved'):activate(layout,x,out,ids,lambda *args:calls.append('c'),lambda *args:calls.append('i'))
check('layout_selects_correct_callback',calls==['c','i'])
for name,layout,xx,oo,ii in [('unknown','bad',x,out,ids),('dtype','canonical',x.float(),out,ids),('shape','canonical',x,out[:2],ids),('ids','canonical',x,out,ids.long()),('stride','canonical',x[:,::2],out[:,:8],ids),('alias','canonical',x[:1],x[:1,:16],ids[:1]),('partial_alias','canonical',x[:1],x.view(-1)[8:24].view(1,16),ids[:1])]:
 rejected=False
 try:activate(layout,xx,oo,ii,lambda *args:None,lambda *args:None)
 except ValueError:rejected=True
 check('reject_'+name,rejected)
if a.cpu:r.update(status='PASS',check_count=len(r['checks']));save();raise SystemExit(0)
for path in (ROOT/'swarm_2026-11-16/moe/humming_vendor',ROOT/'swarm_2026-11-16/moe/phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
from kernel_runtime import bootstrap
bootstrap(ROOT/'sglang')
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.jit.utils import load_jit,make_cpp_args,is_arch_support_pdl,get_activation_cuda_cflags
args=make_cpp_args(torch.bfloat16,is_arch_support_pdl())
phase('load_candidate_native_module')
module=load_jit('research_interleaved_activation',*args,cuda_files=[str(P/'interleaved_activation.cuh')],extra_cuda_cflags=get_activation_cuda_cflags(),cuda_wrappers=[('run_activation_filtered',f'InterleavedActivationKernel<{args}>::run_activation_filtered')])
phase('candidate_native_module_ready')
sp=importlib.util.spec_from_file_location('fallback_scales',ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py');sc=importlib.util.module_from_spec(sp);sp.loader.exec_module(sc)
def canonical(x,out,ids):return silu_and_mul(x,out,expert_ids=ids,expert_step=1)
def interleaved(x,out,ids):return module.run_activation_filtered(x,out,ids,1,'silu')
def exact(a,b):return torch.equal(a.view(torch.int16),b.view(torch.int16))
torch.manual_seed(188);scales=torch.tensor([.25,.75,1.,1.5,2.],device='cuda',dtype=torch.bfloat16)
for rows,hidden in ([(8,4096)] if a.targeted else [(8,4096),(0,64),(1,64),(17,512),(32,4096)]):
 phase('geometry_%d_%d'%(rows,hidden))
 src=torch.randn(rows,2*hidden,device='cuda',dtype=torch.bfloat16);pair=torch.empty_like(src);cid=torch.arange(rows,device='cuda',dtype=torch.int32)%5
 co=torch.full((rows,hidden),37,dtype=torch.bfloat16,device='cuda');io=torch.full_like(co,37)
 def setcase(generation):
  torch.manual_seed(183+generation);src.copy_(torch.randn_like(src)*(generation+1))
  if rows:
   specials=torch.tensor([0.,-0.,torch.finfo(torch.bfloat16).tiny,2**-133,-2**-133,88.,-88.,1.00390625,float('inf'),-float('inf'),float('nan')],device='cuda',dtype=torch.bfloat16)
   if generation==2:src[min(1,rows-1),:specials.numel()].copy_(specials)
   cid.copy_((torch.arange(rows,device='cuda',dtype=torch.int32)+generation)%5)
   if generation==1:cid.fill_(-1)
   elif generation==2:
    cid[::3]=-1
    if rows==1:cid[0]=0
   pair[:,0::2].copy_(src[:,:hidden]);pair[:,1::2].copy_(src[:,hidden:])
 def original():sc.scale_active_rows(src,cid,scales);activate('canonical',src,co,cid,canonical,interleaved)
 def fallback():sc.scale_active_rows(pair,cid,scales);activate('gate_up_interleaved',pair,io,cid,canonical,interleaved)
 setcase(0)
 if rows:
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for warm in range(2):
    phase('warm_original_%d'%warm);original();phase('warm_fallback_%d'%warm);fallback()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  phase('capture_native_and_fallback');ga=torch.cuda.CUDAGraph();gb=torch.cuda.CUDAGraph()
  with torch.cuda.graph(ga):original()
  with torch.cuda.graph(gb):fallback()
 for gen in range(4):
  phase('generation_%d'%gen)
  setcase(gen);co.fill_(37);io.fill_(37)
  if rows:ga.replay();gb.replay()
  else:original();fallback()
  torch.cuda.synchronize();check('rows%d_hidden%d_gen%d_bits'%(rows,hidden,gen),exact(co,io));check('rows%d_gen%d_invalid_untouched'%(rows,gen),bool((io[cid==-1]==37).all()))
  if rows:check('rows%d_gen%d_scaled_input_pair_bits'%(rows,gen),exact(src[:,:hidden],pair[:,0::2].contiguous()) and exact(src[:,hidden:],pair[:,1::2].contiguous()))
 # Distinct pointers use same API, not replaying stale graph addresses.
 if rows:
  setcase(0);xx=src.clone();pp=pair.clone();oo=torch.full_like(co,37);qq=torch.full_like(io,37)
  sc.scale_active_rows(xx,cid,scales);sc.scale_active_rows(pp,cid,scales);activate('canonical',xx,oo,cid,canonical,interleaved);activate('gate_up_interleaved',pp,qq,cid,canonical,interleaved)
  check('changed_pointer_native_rows%d'%rows,exact(oo,qq))
r.update(status='PASS',check_count=len(r['checks']),device=torch.cuda.get_device_name(),compiler_flags=get_activation_cuda_cflags());save()
