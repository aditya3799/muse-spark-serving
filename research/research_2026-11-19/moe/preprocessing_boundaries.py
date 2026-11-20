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
for m in [2047,2048,2049]:
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
report['complete']=True
report['scope']='Only affected M2048 chunk boundaries; earlier exact pack and restoration receipts remain unchanged.'
(here/'normal_package/preprocessing_boundary_2048.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
