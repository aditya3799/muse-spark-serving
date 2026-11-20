"""Isolated scored-helper large-prefill profile. No source modifications or submissions."""
import argparse,ast,gc,hashlib,importlib.util,json,os,pathlib,random,statistics,sys,time
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[3]
EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--out',required=True);pa.add_argument('--tokens',nargs='+',type=int,default=[753,3012,8192,16384]);pa.add_argument('--patterns',nargs='+',default=['uniform','mixed']);pa.add_argument('--samples',type=int,default=8);a=pa.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 assert all(m>512 for m in a.tokens),'This profile deliberately targets scored large-M unsplit path'
 for path in (ROOT/'swarm_2026-11-16/moe/humming_vendor',ROOT/'swarm_2026-11-16/moe/phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming.layer import HummingMethod
 from fixture_tuning_cache import cached_threshold_for_preparation
 import sglang.srt.layers.moe.fused_moe_triton as align_module
 import sglang.kernels.ops.activation.activation as act_module
 scale=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py','sglang.kernels.ops.moe.muse_spark_humming_scale')
 split=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py','sglang.kernels.ops.moe.muse_spark_humming_split')
 final=load(EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py','round3_scored_finalizer')
 src=EXPORT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());names=('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs');nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns)
 testsrc=ROOT/'sglang/python/sglang/test/test_marlin_utils.py';tree=ast.parse(testsrc.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_nvfp4_weight_and_ref');gen={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(testsrc),'exec'),gen)
 report={'status':'PREPARING','scope':'LocalSM86 scored large-prefill helper stage profile. No H200 or serving projection. External CUDA-event nodes instrument a second graph; clean graph measured separately.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),src,testsrc]},'cases':[]}
 def save():
  tmp=out/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'results.json')
 save();torch.manual_seed(771);h=4096;e=1;raw=torch.nn.Module()
 for prefix,n in [('w13',2*h),('w2',h)]:
  weights=[];scales=[];globals_=[]
  for expert in range(e):
   w,s,g,ref=gen[node.name](n,h,torch.bfloat16);weights.append(w);scales.append(s);globals_.append(g);del ref
  for suffix,values in [('weight',weights),('weight_scale',scales),('weight_scale_2',globals_)]:setattr(raw,prefix+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
  del weights,scales,globals_,w,s,g,values
 for stage in ('w13','w2'):getattr(raw,stage+'_weight_scale_2').mul_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16))
 with cached_threshold_for_preparation(out):layer=ns['_prepare_humming_experts'](raw,torch)
 del raw;gc.collect();torch.cuda.empty_cache()
 report.update(status='RUNNING',device=torch.cuda.get_device_name(),SM=torch.cuda.get_device_capability(),weight_allocated_bytes=torch.cuda.memory_allocated());save()
 norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16);flush=torch.zeros(64*1024**2//4,device='cuda',dtype=torch.int32)
 original={'forward':HummingMethod.forward_layer,'align':align_module.moe_align_block_size,'scale':scale.scale_active_rows,'silu':act_module.silu_and_mul}
 trace=None
 def measure(label,fn,*args,**kwargs):
  if trace is None:return fn(*args,**kwargs)
  i=len(trace['entries']);before,after=trace['events'][i];before.record();value=fn(*args,**kwargs);after.record();trace['entries'].append((label,before,after));return value
 def forward(*args,**kwargs):return measure(kwargs.get('sublayer_name','gemm'),original['forward'],*args,**kwargs)
 def align(*args,**kwargs):return measure('alignment',original['align'],*args,**kwargs)
 def scales(*args,**kwargs):return measure('global_scale',original['scale'],*args,**kwargs)
 def silu(*args,**kwargs):return measure('silu',original['silu'],*args,**kwargs)
 HummingMethod.forward_layer=forward;align_module.moe_align_block_size=align;scale.scale_active_rows=scales;act_module.silu_and_mul=silu
 def cleanup():gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()

 import torch.cuda.profiler
 m=753
 free,total=torch.cuda.mem_get_info()
 if free < 512*1024**2+256*1024**2:
  report.update(status='HELD_MEMORY_RESERVE',free_bytes=free);save();return
 x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20
 ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32);ids[:,0]=0
 def run():
  return ns['_humming_expert_outputs'](layer,x,ids,torch,defer_downscale=True,global_experts=256)
 for _ in range(2):
  v=run();del v
 torch.cuda.synchronize()
 def ranged(*args,**kwargs):
  label='okc_'+kwargs.get('sublayer_name','gemm')
  torch.cuda.nvtx.range_push(label)
  try:return original['forward'](*args,**kwargs)
  finally:torch.cuda.nvtx.range_pop()
 HummingMethod.forward_layer=ranged
 torch.cuda.profiler.start()
 value,local=run()
 torch.cuda.synchronize()
 torch.cuda.profiler.stop()
 assert torch.isfinite(value[:,0]).all()
 report.update(status='COMPLETE_SINGLE_EXPERT_COUNTER_PROBE',scope='E1 M753 all rows routed once to local expert; ordinary helper warmup then one profiled call. Not H200 or serving evidence. TTS unchanged.',peak_allocated=torch.cuda.max_memory_allocated(),selected_configs=layer.fp32_configs)
 save()
if __name__=='__main__':main()
