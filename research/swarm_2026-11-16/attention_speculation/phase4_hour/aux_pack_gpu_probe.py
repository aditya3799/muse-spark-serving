"""Bounded direct-cast auxiliary pack CUDA graph contract, no model execution."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch
from aux_pack_contract import make_aux_owner
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
r={'status':'running','cases':[],'scope':'Isolated FP32->BF16 cast/pack + shared graph output only; no model integration, no H200 memory or performance evidence.'}
def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
try:
 torch.manual_seed(6217);h=128;maxn=512;ids=(1,16,30,44,59)
 owner=make_aux_owner(torch,max_tokens=maxn,hidden=h,capture_ids=ids,num_layers=62,device='cuda')
 r['device']=torch.cuda.get_device_name();pointer=owner.buffer.data_ptr();graphs={};inputs={};refs={}
 def make_inputs(n):return [torch.empty((n,h),device='cuda',dtype=torch.float32) for _ in ids]
 def populate(streams,seed):
  torch.manual_seed(seed)
  for x in streams:
   bits=torch.randint(-(1<<31),(1<<31)-1,x.shape,device='cuda',dtype=torch.int32)
   special=torch.tensor([0,-2147483648,1,8388607,2139095040,-8388608,2143289345,2147483647],device='cuda',dtype=torch.int32)
   bits.view(-1)[:8].copy_(special);x.copy_(bits.view(torch.float32))
 for n in (512,256,128,32,8):
  streams=make_inputs(n);populate(streams,100+n);inputs[n]=streams
  def candidate():
   for i,x in enumerate(streams):owner.buffer[:n,i*h:(i+1)*h].copy_(x)
   return owner.buffer[:n]
  def original():return torch.cat([x.to(torch.bfloat16) for x in streams],dim=-1)
  for _ in range(2):candidate();original()
  torch.cuda.synchronize();gc=torch.cuda.CUDAGraph();gb=torch.cuda.CUDAGraph()
  with torch.cuda.graph(gc):candidate()
  with torch.cuda.graph(gb):ref=original()
  graphs[n]=(gc,gb);refs[n]=ref
 for step,n in enumerate((512,8,128,32,256,512,32,8,256,128)):
  lease=owner.begin(n);populate(inputs[n],301+step)
  try:owner.begin(n)
  except RuntimeError:pass
  else:raise AssertionError('Consumer ownership was not enforced')
  owner.buffer.fill_(-3.0)
  gc,gb=graphs[n];gc.replay();gb.replay()
  out=owner.buffer[:n];ref=refs[n]
  equal=torch.equal(out.view(torch.int16),ref.view(torch.int16))
  r['cases'].append({'tokens':n,'step':step,'bitwise_equal':equal,'source_changed':True,'stable_pointer':out.data_ptr()==pointer})
  assert equal,'Cast/pack differs from stock to+cat'
  assert out.data_ptr()==pointer
  assert bool((owner.buffer[n:]==-3.0).all()),'Padded output rows overwritten'
  assert owner.adopt_bcg_output(out.clone(),maxn) is None
  assert owner.adopt_bcg_output(owner.buffer[1:],maxn) is None
  assert owner.adopt_bcg_output(out,0) is None
  assert owner.adopt_bcg_output(out,maxn+1) is None
  adopted=owner.adopt_bcg_output(out,maxn);assert adopted.data_ptr()==pointer
  # Consumer owns contents until finish; changing all source streams cannot
  # change already captured output, proving no borrowed-view alias.
  saved=out.clone()
  for x in inputs[n]:x.zero_()
  assert torch.equal(saved.view(torch.int16),out.view(torch.int16))
  owner.release_after_materialization(lease.generation);save()
 r.update(status='PASS',checks=len(r['cases']),captures=10,shared_output_bytes=owner.buffer.numel()*owner.buffer.element_size(),source_sha256=hashlib.sha256((P/'aux_pack_contract.py').read_bytes()).hexdigest());save();print(json.dumps(r))
except Exception as e:
 r.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());save();raise
