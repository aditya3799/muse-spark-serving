"""Shipped preprocessing CUDA gate; no Hopper GEMM or serving speed claim."""
from pathlib import Path
import sys,importlib.util,json,hashlib
import numpy as np
import torch,triton
import triton.language as tl
here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/'research_2026-11-18_rl/expert_slab'))
from test_slab import pack,scales
file=here/'normal_package/muse_spark_grouped_prep.py'
spec=importlib.util.spec_from_file_location('shipped_group_prep',file)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

@triton.jit
def restore_native(src,dst,N:tl.constexpr):
 i=tl.program_id(0)*256+tl.arange(0,256)
 word=tl.load(src+i,i<N,other=0)
 out=tl.inline_asm_elementwise('{ .reg .b32 s,c; mov.b32 s,0x40804080; mov.b32 c,0x80008000; fma.rn.bf16x2 $0,$1,s,c; }',constraints='=r,r',args=[word],dtype=tl.uint32,is_pure=True,pack=1)
 tl.store(dst+i,out,i<N)

@triton.jit
def restore_current(src,dst,N:tl.constexpr):
 i=tl.program_id(0)*256+tl.arange(0,256)
 value=tl.load(src+i,i<N,other=0)
 tl.store(dst+i,value.to(tl.float32)*4.,i<N)

rng=np.random.default_rng(917443);report=dict(device=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),pack=[],routing=[],complete=False)
for e,n,k in [(3,128,192),(2,192,128)]:
 q=rng.integers(0,16,(e,n,k),dtype=np.uint8)
 sf=rng.integers(0,256,(e,n,k//16),dtype=np.uint8)
 sh=scales(torch.from_numpy(sf)).cuda()
 chosen=[e-1,0];chosen_gpu=torch.tensor(chosen,device='cuda',dtype=torch.int32)
 expect_q=q[chosen,:,::2]|(q[chosen,:,1::2]<<4)
 words=sf[chosen].transpose(0,2,1).reshape(2,k//64,4,n).astype(np.uint32)
 expect_s=np.bitwise_or.reduce(words<<(np.arange(4,dtype=np.uint32)[None,None,:,None]*8),axis=2)
 storage=torch.full((2*n*k*9//16+32,),0x55,device='cuda',dtype=torch.uint8)
 for wg in [False,True]:
  a,b=p.convert(pack(q,wg).cuda(),sh,chosen_gpu,storage,n,k,wg)
  assert np.array_equal(a.cpu().numpy(),expect_q)
  assert np.array_equal(b.cpu().numpy().view(np.uint32),expect_s)
  assert bool(torch.all(storage[-32:]==0x55))
  report['pack'].append(dict(e=e,n=n,k=k,wgmma=wg,nibbles=a.numel()*2,scale_bytes=b.numel()*4,exact=True,tail_canary=True))
for m in [3071,3072,3073]:
 e=32;h=64
 ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32)
 x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16)
 grouped=torch.empty((m*8+e*16,h),device='cuda',dtype=torch.bfloat16)
 out=torch.empty((m*8,h),device='cuda',dtype=torch.bfloat16)
 bufs=p.route_buffers(ids,e)
 def call():
  grouped.fill_(float('nan'))
  mapping=p.route(ids,bufs);p.gather(x,grouped,mapping);p.scatter(grouped,out,mapping[0])
  return mapping
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  call();call();stream.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):mapping=call()
 torch.cuda.current_stream().wait_stream(stream)
 for iteration,mode in enumerate(['uniform','skew','none','uniform']):
  keys=rng.integers(-1,e,(m,8),dtype=np.int32)
  if mode=='skew':keys[:,:6]=0
  if mode=='none':keys.fill(-1)
  ids.copy_(torch.from_numpy(keys));x.copy_(torch.arange(m*h).reshape(m,h)%511+iteration)
  graph.replay();torch.cuda.synchronize()
  gold=x.repeat_interleave(8,0).mul_(4);gold[(ids.flatten()<0)|(ids.flatten()>=e)]=0
  assert torch.equal(out.view(torch.int16),gold.view(torch.int16))
  dest,base,count,padded=[z.cpu().numpy() for z in mapping]
  expected=np.full(m*8,-1,np.int32);offset=0
  for j in range(e):
   slots=np.flatnonzero(keys.flatten()==j);length=max(16,(len(slots)+15)//16*16)
   assert (base[j],count[j],padded[j])==(offset,len(slots),length)
   expected[slots]=np.arange(offset,offset+len(slots));offset+=length
  assert np.array_equal(dest,expected) and offset<=m*8+16*e
  report['routing'].append(dict(m=m,mode=mode,graph=True,exact=True,values=out.numel()))
values=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
a=torch.empty_like(values);b=torch.empty_like(values)
restore_native[(128,)](values.view(torch.uint32),a.view(torch.uint32),32768)
restore_current[(256,)](values,b,65536)
finite_or_inf=(values.view(torch.int16).to(torch.int32)&0x7fff)<=0x7f80
bad=int(((a.view(torch.int16)!=b.view(torch.int16))&finite_or_inf).sum())
report['restore_all_bf16']=dict(values=int(finite_or_inf.sum()),unequal_bits=bad,includes_subnormals_and_infinities=True)
report['complete']=True
(here/'normal_package/preprocessing_gpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
assert bad==0,bad
