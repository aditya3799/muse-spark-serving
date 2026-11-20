"""Full40960→8192 same-M auxiliary staging gate; no target model/perf claim."""
from __future__ import annotations
import argparse,ast,gc,hashlib,inspect,json,time
from pathlib import Path
from types import SimpleNamespace as N
import torch
from torch import nn
import prefix_width_scaffold as loader
P=Path(__file__).parent;ROOT=P.parents[3];SRT=ROOT/'sglang/python/sglang/srt'
torch.set_num_threads(1)
BUDGET=900<<20;RESERVE=512<<20

def node(path,cls,name):
 t=ast.parse(path.read_text());owner=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls)
 return next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)

def main(out):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 report={'status':'STARTED','checks':[],'budget_bytes':BUDGET,'reserve_bytes':RESERVE,'external_tts':'Untouched; any timing invalid for performanceclaim','sources':{}}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def phase(name):report['phase']=name;save();print(name,flush=True)
 def memory(initial=False):
  free,total=torch.cuda.mem_get_info();allocated=torch.cuda.memory_allocated();peak=torch.cuda.max_memory_allocated()
  report.setdefault('memory',[]).append(dict(free=free,allocated=allocated,peak=peak))
  assert peak<=BUDGET,'900MiB allocation budget exceeded'
  need=RESERVE+max(0,BUDGET-allocated)
  if free<need:
   report['status']='SKIPPED_MEMORY_RESERVE';save();raise RuntimeError('Insufficient free memory for reserve')
 def check(name,a,b):
  finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
  bits=torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
  row={'name':name,'finite':finite,'bitwise':bits,'elements':a.numel()};report['checks'].append(row)
  if not(finite and bits):
   row['different_elements']=int((a!=b).sum());row['maxabs']=float((a.float()-b.float()).abs().max())
   report['status']='FAIL_NUMERICAL';save();raise AssertionError(name)
 save();assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats();memory(True)
 import sgl_kernel,importlib.metadata
 report['sgl_kernel_version']=getattr(sgl_kernel,'__version__','unknown')
 report['distribution_metadata']='sgl-kernel dist-info unavailable; moduleversion and loadedbinary hashes used'
 nativefile=getattr(getattr(sgl_kernel,'common_ops',None),'__file__',None)
 if nativefile and Path(nativefile).is_file():report['sources'][nativefile]=hashlib.sha256(Path(nativefile).read_bytes()).hexdigest()
 normpath=Path(inspect.getfile(sgl_kernel.rmsnorm));report['sources'][str(normpath)]=hashlib.sha256(normpath.read_bytes()).hexdigest()
 ns={'torch':torch,'nn':nn,'rmsnorm':sgl_kernel.rmsnorm,'is_batch_invariant_mode_enabled':lambda:False,'_flashinfer_rmsnorm_quant_available':False}
 modelpath=SRT/'models/dflash.py';normsource=SRT/'layers/layernorm.py'
 loader.load([node(modelpath,'DFlashDraftModel','project_target_hidden'),node(normsource,'RMSNorm','forward_cuda')],ns)
 Norm=type('ExtractedCUDA_RMS',(),{'__call__':ns['forward_cuda']});Model=type('ExtractedDraft',(),{'project_target_hidden':ns['project_target_hidden']})
 model=Model();model.quantizable=False;model.num_context_features=5;model.config=N(hidden_size=8192)
 phase('ALLOCATE_SHARED_WEIGHT')
 class ObservedLinear(nn.Linear):
  def forward(self,x):
   self.last=super().forward(x);return self.last
 model.fc=ObservedLinear(40960,8192,bias=False,dtype=torch.bfloat16,device='cuda')
 model.fc.weight.requires_grad_(False)
 norm=Norm();norm._quant_linear=None;norm.variance_size_override=None;norm.cast_x_before_out_mul=False
 norm.weight=torch.linspace(.5,1.5,8192,device='cuda',dtype=torch.bfloat16);norm.variance_epsilon=1e-6
 model.hidden_norm=norm
 report['matmul_settings']={'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
 report['norm_branch']='Actual extracted RMSNorm.forward_cuda standard residual=None/cast=False branch calls installed sgl_kernel.rmsnorm'
 def capture(call):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):call()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph): result=call()
  return graph,result
 with torch.inference_mode():
  for bs in (1,3,8,32):
   memory();phase('SHAPE_'+str(bs))
   oracle=torch.empty((bs,8,40960),dtype=torch.bfloat16,device='cuda').normal_(std=.125)
   first=torch.empty((bs,4,40960),dtype=torch.bfloat16,device='cuda')
   stage=torch.empty_like(oracle);commit=torch.ones(bs,dtype=torch.int32,device='cuda')
   assert len({oracle.data_ptr(),first.data_ptr(),stage.data_ptr()})==3
   def baseline():
    normalized=model.project_target_hidden(oracle.view(bs*8,40960));return model.fc.last,normalized
   def candidate():
    stage.zero_();first.copy_(oracle[:,:4]);stage[:,:4].copy_(first)
    first.fill_(float('nan')) # poison shared first-target output after safecopy
    stage[:,4:].copy_(torch.where((commit>4)[:,None,None],oracle[:,4:],0))
    normalized=model.project_target_hidden(stage.view(bs*8,40960));return model.fc.last,normalized
   ga,a=capture(baseline);gb,b=capture(candidate)
   assert a[0].data_ptr()!=b[0].data_ptr() and a[1].data_ptr()!=b[1].data_ptr()
   input_fingerprints=[(x,x.data_ptr(),tuple(x.shape),tuple(x.stride())) for x in (oracle,first,stage,commit,model.fc.weight,norm.weight)]
   for generation in range(8):
    memory();lens=(torch.arange(bs,dtype=torch.int32)+generation)%8+1
    commit.copy_(lens);oracle.normal_(std=.125*(1+generation%3))
    old_fc=model.fc.weight[0,0].clone();old_norm=norm.weight.clone()
    model.fc.weight[0,0].add_(.0001220703125)
    norm.weight.add_(.015625 if generation%2==0 else -.015625)
    assert not torch.equal(old_fc,model.fc.weight[0,0]) and not torch.equal(old_norm,norm.weight)
    report['parameter_mutations_checked']=report.get('parameter_mutations_checked',0)+1
    del old_fc,old_norm
    assert all(x.data_ptr()==ptr and tuple(x.shape)==shape and tuple(x.stride())==stride for x,ptr,shape,stride in input_fingerprints)
    ga.replay();gb.replay();torch.cuda.synchronize()
    valid=(torch.arange(8)[None,:]<lens[:,None]).flatten().cuda()
    check(f'B{bs}_G{generation}_rawFC',a[0][valid],b[0][valid])
    check(f'B{bs}_G{generation}_nativeRMS',a[1][valid],b[1][valid])
    assert bool(torch.isnan(first).all())
    assert bool(torch.isfinite(b[0]).all() and torch.isfinite(b[1]).all())
    report.setdefault('covered_commit_lengths',[]).extend(lens.unique().tolist())
   save();del ga,gb,a,b,oracle,first,stage,commit,input_fingerprints,valid
   model.fc.last=None;gc.collect();torch.cuda.empty_cache();memory()
 memory();report.update(status='PASS_FULLSHAPE_STAGING_COMPONENT_ONLY',elapsed_s=time.monotonic()-start,
  peak_allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
  covered_commit_lengths=sorted(set(report['covered_commit_lengths'])),device=torch.cuda.get_device_name(),
  limits=['No target model/FA3 attention or segmentedverification arithmetic','No full five-layer GPU KV append; CPU reducedgeometry append separatelytested',
   'Native installed sgl_kernel branch, not claim of binary identity withjudgeimage','Same full8B FC retained; no performance or shape-speed claim',
   'GPU staging uses fixedcapacity where-mask for numericalfixture, not a production survivor dispatcher'])
 for path in (modelpath,normsource,P/'prefix_width_scaffold.py',Path(__file__)):
  report['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 save();print(json.dumps({k:v for k,v in report.items() if k not in ('checks','memory')},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)