"""Actual frozen/patched sampler classes; synthetic dense heads and virtual TP only."""
import ast,hashlib,json,sys,types
from pathlib import Path
import torch
from packed_sampler import PackedDraftCollective
here=Path(__file__).resolve().parent

def load_class(path,name):
 tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='_DflashDraftSampler')
 env=dict(torch=torch);exec(compile(ast.Module(body=[cls],type_ignores=[]),name,'exec'),env);return env[cls.name]

class Peers:
 def __init__(self,world,compact=None):self.world_size=world;self.compact=compact;self.maxima=None;self.ids=None;self.payload=None;self.calls=0
 def all_gather_into_tensor(self,out,local):
  self.calls+=1
  if self.compact is None:source=self.ids if local.dtype==torch.int64 else self.maxima
  else:source=self.payload
  out.copy_(source.flatten());out[:local.numel()].copy_(local)

@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.manual_seed(1739517)
 # Only constructor transport-policy acquisition is injected. It has separate
 # realGloo agreement tests; all class arithmetic and helper kernels are actual.
 module=types.ModuleType('sglang.kernels.ops.speculative.packed_draft_sampler');module.maybe_prepare_packed=lambda **kwargs:None
 sys.modules[module.__name__]=module
 original=load_class(here/'dflash_sources/dflash_worker_v2.py','frozen_sampler')
 modified=load_class(here/'sampler_integration/sources/python/sglang/srt/speculative/dflash_worker_v2.py','patched_sampler')
 records=[]
 for dtype in (torch.bfloat16,torch.float16,torch.float32):
  bitdtype=torch.int32 if dtype==torch.float32 else torch.int16
  for world in (2,8):
   widths=[37-rank for rank in range(world)];offsets=[sum(widths[:rank]) for rank in range(world)]
   weights=[torch.randn((width+5,256),device='cuda',dtype=dtype) for width in widths]
   for bs in (1,4,8):
    n=bs*7;hidden=torch.randn((bs*8,256),device='cuda',dtype=dtype);input_ids=torch.zeros(bs*8,device='cuda',dtype=torch.int64)
    maxima=torch.empty((world,n),device='cuda',dtype=dtype);ids=torch.empty((world,n),device='cuda',dtype=torch.int64)
    peer=Peers(world);peer.maxima=maxima;peer.ids=ids
    native=original(weight=weights[0],block_size=8,num_org=widths[0],org_vocab_start=offsets[0],max_bs=8,tp_group=peer)
    candidates=[]
    for compact in (False,True):
     group=Peers(world,compact);group.payload=torch.empty((world,n) if compact else (world,n,2),device='cuda',dtype=torch.int64)
     sampler=modified(weight=weights[0],block_size=8,num_org=widths[0],org_vocab_start=offsets[0],max_bs=8,tp_group=group)
     sampler._packed_argmax=PackedDraftCollective(tp_size=world,max_tokens=56,dtype=dtype,device=weights[0].device,compact=compact)
     candidates.append((sampler,group))
    def stage(kind):
     hidden.normal_();input_ids.random_(0,4096)
     if kind=='integer':hidden.copy_(torch.randint(-4,5,hidden.shape,device='cuda').to(dtype))
     if kind=='all_tie':hidden.zero_()
     if kind=='nan':hidden[3::8,0]=float('nan')
     hs=hidden.view(bs,8,256)[:,1:,:].reshape(n,256);all_logits=[]
     for rank in range(world):
      logits=torch.matmul(hs,weights[rank][:widths[rank]].T);all_logits.append(logits)
      value,index=torch.max(logits,dim=-1);maxima[rank].copy_(value);ids[rank].copy_(index+offsets[rank])
     for sampler,group in candidates:
      if group.compact:group.payload.copy_(((maxima.view(bitdtype).to(torch.int64)&(0xffffffff if dtype==torch.float32 else 0xffff))<<32)|ids)
      else:group.payload[:,:,0].copy_(maxima.view(bitdtype));group.payload[:,:,1].copy_(ids)
     return torch.cat(all_logits,dim=-1).argmax(-1)
    expected=stage('normal');native(hidden,input_ids)
    for sampler,group in candidates:sampler(hidden,input_ids)
    torch.cuda.synchronize();graphs=[]
    for sampler in [native]+[x[0] for x in candidates]:
     with torch.cuda.graph(graph:=torch.cuda.CUDAGraph()):sampler(hidden,input_ids)
     graphs.append(graph)
    for kind in ('normal','integer','all_tie','nan'):
     expected=stage(kind)
     for graph in graphs:graph.replay()
     torch.cuda.synchronize()
     assert torch.equal(native.out[:n],expected)
     for sampler,group in candidates:
      assert torch.equal(sampler.out[:n],native.out[:n]),(dtype,world,bs,kind,group.compact)
      records.append(dict(dtype=str(dtype),world=world,B=bs,compact=group.compact,input=kind,tokens_exact=True,full_vocab_argmax_exact=True))
    print('PASS actualclass',dtype,'TP',world,'B',bs,flush=True)
 from sampler_edge_controls import run as edge_controls
 edges=edge_controls()
 doc=dict(scope=__doc__,passed=True,records=records,edge_controls=edges,constructor_policy='injected None; separatelytestedrealGloo',transport='virtualpeer D2D copies, noNCCLtime',
  source_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['packed_sampler.py','packed_sampler_kernel.py','sampler_class_probe.py','sampler_edge_controls.py']})
 (here/'sampler_class_probe.json').write_text(json.dumps(doc,indent=2))
if __name__=='__main__':main()
