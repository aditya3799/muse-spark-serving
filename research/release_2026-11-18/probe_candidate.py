"""Bounded integrated partition-only correctness; no serving/timing claim."""
from pathlib import Path
import argparse,ast,copy,gc,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent;ROOT=P.parent
OLD=ROOT/'swarm_2026-11-16/moe'
SCORED=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'
CAND=P/'candidate_sources/python/sglang'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=True);assert not(a.out/'results.json').exists()
 report={'status':'STARTED','scope':'Integrated package SM86 H=I256,E32,top8,M513/768. Exact scored reference vs actual new preparation/selector/full helper. No H200/full-model/multirank/timing claim.','memory':[],'checks':[],'cases':[],'plans':[]}
 def save():(a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 save()
 try:
  manifest=json.loads((P/'BUILD_MANIFEST.json').read_text())
  assert sha(P/'candidate.diff')==manifest['patch_sha256']
  for name,digest in manifest['source_hashes'].items():assert sha(P/'candidate_sources'/name)==digest,name
  report['package_sha256']=manifest['patch_sha256']
  for d in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(d))
  from kernel_runtime import bootstrap
  bootstrap(ROOT/'sglang')
  import torch
  from fixture_tuning_cache import cached_threshold_for_preparation
  assert torch.cuda.get_device_capability()==(8,6),'local qualification scope only'
  budget=128*1024**2;reserve=512*1024**2;entry=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
  def memory(label,extra=0):
   free,_=torch.cuda.mem_get_info();allocated=torch.cuda.memory_allocated()-entry;peak=torch.cuda.max_memory_allocated()-entry
   row=dict(label=label,free=free,allocated_delta=allocated,peak_delta=peak,estimated_next=extra);report['memory'].append(row);save()
   assert free>=reserve+extra and peak<=budget and allocated+extra<=budget,row
  memory('entry',32*1024**2)
  def load(name,path):
   spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
  scale=load('sglang.kernels.ops.moe.muse_spark_humming_scale',SCORED/'kernels/ops/moe/muse_spark_humming_scale.py')
  oldsplit=load('release_scored_split',SCORED/'kernels/ops/moe/muse_spark_humming_split.py')
  new=load('sglang.kernels.ops.moe.muse_spark_humming_split',CAND/'kernels/ops/moe/muse_spark_humming_split.py')
  final=load('release_scored_finalizer',SCORED/'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
  def extract(path):
   names=('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')
   tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
   assert len(nodes)==3;ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
  oldns=extract(SCORED/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py')
  newns=extract(CAND/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py')
  deps=[Path(__file__),P/'candidate.diff',P/'BUILD_MANIFEST.json',*sorted((P/'candidate_sources').rglob('*.py')),SCORED/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py',Path(oldsplit.__file__),Path(scale.__file__),Path(final.__file__),ROOT/'optimization_marlin_tiles_2026-11-15/kernel_runtime.py',OLD/'phase2/fixture_tuning_cache.py',ROOT/'sglang/python/sglang/kernels/ops/activation/activation.py',ROOT/'sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh',*sorted((OLD/'humming_vendor/humming').rglob('*.py')),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
  snapshot=a.out/'dependencies';snapshot.mkdir(exist_ok=True)
  report['source_hashes']={}
  for source in deps:
   assert source.exists(),str(source)
   rel=source.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes());report['source_hashes'][str(rel)]=sha(source)
  save()
  torch.manual_seed(31991);h=256;e=32;topk=8;raw=torch.nn.Module()
  for stage,n in [('w13',2*h),('w2',h)]:
   w=torch.randint(0,256,(e,n,h//2),device='cuda',dtype=torch.uint8)
   sf=torch.randint(40,81,(e,n,h//16),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
   gs=torch.linspace(.125,.5,e,device='cuda',dtype=torch.bfloat16)
   for suffix,t in [('weight',w),('weight_scale',sf),('weight_scale_2',gs)]:setattr(raw,stage+'_'+suffix,torch.nn.Parameter(t,requires_grad=False))
  memory('before_preparation',24*1024**2)
  # Both functions execute their real source bodies; candidate includes actual setup hook.
  with cached_threshold_for_preparation(a.out):
   base=oldns['_prepare_humming_experts'](raw,torch)
   cand=newns['_prepare_humming_experts'](raw,torch)
  assert getattr(cand,'_humming_large_prefill_partition_enabled',False) is True,'new setup did not enable'
  assert type(cand._humming_large_prefill_configs) is frozenset and cand._humming_large_prefill_configs
  assert not getattr(base,'_humming_large_prefill_partition_enabled',False)
  assert base.fp32_configs==cand.fp32_configs
  for stage in ('w13','w2'):
   for suffix in ('weight','weight_scale','post_scale'):
    aa=getattr(base,stage+'_'+suffix);bb=getattr(cand,stage+'_'+suffix)
    assert torch.equal(aa.view(torch.uint8),bb.view(torch.uint8)),(stage,suffix)
  del raw,w,sf,gs,t,aa,bb;gc.collect();torch.cuda.empty_cache()
  report['admitted_signatures']=len(cand._humming_large_prefill_configs);report['owned_configs']=cand.fp32_configs;save()
  norm=torch.linspace(.75,1.25,h,device='cuda',dtype=torch.bfloat16)
  original_partition=new.partition_routes_scalable;observed=[]
  def observe_partition(ids,experts,cutoff):
   observed.append({'routes':ids.numel(),'experts':experts,'cutoff':cutoff})
   return original_partition(ids,experts,cutoff)
  new.partition_routes_scalable=observe_partition
  def compare(actual,expected,label):
   finite=bool(torch.isfinite(actual).all()and torch.isfinite(expected).all());eq=torch.equal(actual,expected)
   max_abs=float((actual.float()-expected.float()).abs().max()) if actual.numel()else 0.
   report['checks'].append(dict(label=label,finite=finite,bitwise=eq,max_abs=max_abs,elements=actual.numel()));save()
   assert finite and eq,label
  for m in (513,768):
   plan=new._choose_large_prefill_configs(cand,m*topk,256)
   assert plan is not None,('FALLBACK_NOT_OPTIMIZATION_PASS',m)
   report['plans'].append({'m':m,'plan':plan});save()
   x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,topk),device='cuda',dtype=torch.int32);scores=torch.empty((m,topk),device='cuda')
   def inputs(pattern,generation):
    gen=torch.Generator(device='cpu');gen.manual_seed(912+m+generation)
    rows=[]
    for row in range(m):
     if pattern=='hot' or(pattern=='mixed'and row%2==0):global_ids=list(range(8))
     elif pattern=='remote':global_ids=list(range(32,40))
     else:global_ids=torch.randperm(256,generator=gen)[:8].tolist()
     assert len(set(global_ids))==8
     rows.append([i if i<32 else -1 for i in global_ids])
    cpu=torch.tensor(rows,dtype=torch.int32);ids.copy_(cpu)
    torch.manual_seed(774+generation);x.copy_(torch.randn_like(x)*.125);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
    return cpu
   for defer in (False,True):
    inputs('uniform',0)
    def run(layer,namespace,reference=False):
     if reference:
      down,_=oldsplit.split_expert_outputs(namespace['_humming_expert_outputs_default'],layer,x,ids,torch,global_experts=256,defer_downscale=defer)
     else:down,_=namespace['_humming_expert_outputs'](layer,x,ids,torch,global_experts=256,defer_downscale=defer)
     kw={'expert_output_scale':layer.w2_post_scale}if defer else{}
     out=final.fused_post_expert_norm_weighted_sum(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,**kw)
     return down,out
    graphs=[];outputs=[];before_observed=len(observed)
    for layer,namespace,reference in ((base,oldns,True),(cand,newns,False)):
     memory('before_capture',24*1024**2)
     for _ in range(2):run(layer,namespace,reference)
     torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):out=run(layer,namespace,reference)
     graphs.append(graph);outputs.append(out);memory('after_capture')
    assert len(observed)>before_observed,'candidate never called scalable histogram'
    assert outputs[0][0].data_ptr()!=outputs[1][0].data_ptr() and outputs[0][1].data_ptr()!=outputs[1][1].data_ptr()
    for generation in range(2):
     for pattern in ('uniform','hot','mixed','remote'):
      cpu=inputs(pattern,generation)
      for values in outputs:
       for tensor in values:tensor.fill_(float('nan'))
      for graph in graphs:graph.replay()
      torch.cuda.synchronize();active=ids>=0
      if bool(active.any()):compare(outputs[1][0][active],outputs[0][0][active],[m,defer,generation,pattern,'raw'])
      compare(outputs[1][1],outputs[0][1],[m,defer,generation,pattern,'final'])
      inactive=~active.any(-1);assert bool((outputs[1][1][inactive]==0).all())
      report['cases'].append({'m':m,'defer_downscale':defer,'generation':generation,'pattern':pattern,'local_counts':torch.bincount(cpu[cpu>=0].to(torch.int64),minlength=e).tolist()});memory('case_complete')
    del graph,out,values,tensor,graphs,outputs,active,layer,namespace;gc.collect();torch.cuda.empty_cache();memory('graphs_released')
   del x,ids,scores;gc.collect();torch.cuda.empty_cache()
  new.partition_routes_scalable=original_partition
  report['scalable_partition_observed_calls']=observed
  assert all(sha(ROOT/path)==digest for path,digest in report['source_hashes'].items()),'source changed during probe'
  report['status']='PASS_INTEGRATED_NATIVE_PARTITION_ONLY_HELPER';save();print(json.dumps({'status':report['status'],'cases':len(report['cases']),'checks':len(report['checks'])}))
 except Exception as error:
  report['status']='FAILED';report['error']=repr(error);save();raise
if __name__=='__main__':main()
