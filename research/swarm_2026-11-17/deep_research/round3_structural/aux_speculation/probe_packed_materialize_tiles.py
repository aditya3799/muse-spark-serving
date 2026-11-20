"""Bounded exact-body tile resource pilot; no generic/H200/serving claim."""
import argparse,ast,gc,hashlib,importlib.util,json,random,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for d in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(d))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch,triton
 from fixture_tuning_cache import cached_threshold_for_preparation
 from adversarial_fixture import weight_fixture
 from packed_staged_ops import packed_dequant_kernel
 props=torch.cuda.get_device_properties();assert(props.major,props.minor)==(8,6)
 r={'status':'RUNNING','scope':'Same exact packed_dequant_kernel body, quartervalue/scales/layout unchanged. E1W13/W2 materialization only; localSM86 noH200/genericE32/servingclaim. M64nativeB16GEMMs~1.62ms alreadyexceed~.646mswholebaseline, so evenfreeconversioncannotwinM64geometry.','checks':[],'kernels':[],'timings':[],'environment':'ExternalTTSserviceunchanged; alltimingsprovisional','source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'packed_staged_ops.py',P/'adversarial_fixture.py',Path(__file__))}}
 def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 free,total=torch.cuda.mem_get_info();r['preflight']={'free':free,'estimated_additional':640*1024**2,'reserve':512*1024**2};save()
 if free<(640+512)*1024**2:r['status']='MEMORY_PREFLIGHT_BLOCKED';save();return
 src=EXPORT/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_humming_experts');ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),ns)
 raw=torch.nn.Module();h=4096
 for stage,n in (('w13',8192),('w2',4096)):
  w,s,g=weight_fixture(n,h,stage,0,0,device='cuda')
  for suffix,t in (('weight',w),('weight_scale',s),('weight_scale_2',g)):setattr(raw,stage+'_'+suffix,torch.nn.Parameter(t.unsqueeze(0),requires_grad=False))
 with cached_threshold_for_preparation(a.out):layer=ns['_prepare_humming_experts'](raw,torch)
 del w,s,g,t
 reference=torch.empty((8192,h),dtype=torch.bfloat16,device='cuda');candidate=torch.empty_like(reference);flush=torch.empty(64*1024**2,dtype=torch.uint8,device='cuda')
 variants=[(block,warps) for block in (256,512,1024) for warps in (4,8)]
 def exact(x,y,**meta):
  xx=x.flatten();yy=y.flatten();equal=True;finite=True
  for lo in range(0,xx.numel(),1<<20):
   u=xx[lo:lo+(1<<20)];v=yy[lo:lo+(1<<20)];equal=equal and torch.equal(u.view(torch.int16),v.view(torch.int16));finite=finite and bool(torch.isfinite(u).all() and torch.isfinite(v).all())
   if not equal:
    where=(u.view(torch.int16)!=v.view(torch.int16)).nonzero().flatten()[:16].tolist();meta['first_mismatch']=[{'index':lo+i,'candidate_bits':int(u.view(torch.int16)[i]),'reference_bits':int(v.view(torch.int16)[i])} for i in where];break
  r['checks'].append(dict(meta,bitwise=equal,finite=finite,elements=xx.numel()));save()
  if not(equal and finite):r['status']='FAILED_STRICT_BITWISE_GATE';save();raise AssertionError(meta)
 def sample(g,cold):
  if cold:flush.fill_(91)
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();g.replay();end.record();end.synchronize();return begin.elapsed_time(end)*1000
 for stage,n in (('w13',8192),('w2',4096)):
  graphs={};views={}
  for block,warps in [(4096,4)]+variants:
   key=f'{block}_{warps}';out=reference[:n] if block==4096 else candidate[:n];views[key]=out
   def call():return packed_dequant_kernel[(triton.cdiv(n*h,block),)](getattr(layer,stage+'_weight')[0].view(torch.uint8),getattr(layer,stage+'_weight_scale')[0].view(torch.uint8),out,n,h,False,False,block,num_warps=warps)
   kernel=call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):call()
   graphs[key]=g
   r['kernels'].append({'stage':stage,'block':block,'warps':warps,'n_regs':kernel.n_regs,'n_spills':kernel.n_spills,'shared':kernel.metadata.shared,'kernel_name':kernel.name,'kernel_hash':kernel.hash});save()
  for generation in range(3):
   if generation:
    w,s,g=weight_fixture(n,h,stage,0,generation,device='cuda');getattr(raw,stage+'_weight')[0].copy_(w);getattr(raw,stage+'_weight_scale')[0].copy_(s);del w,s,g
    fresh=ns['_prepare_humming_experts'](raw,torch)
    for suffix in ('weight','weight_scale'):getattr(layer,stage+'_'+suffix).copy_(getattr(fresh,stage+'_'+suffix))
    del fresh
   graphs['4096_4'].replay();torch.cuda.synchronize()
   for block,warps in variants:
    candidate.fill_(37.);graphs[f'{block}_{warps}'].replay();torch.cuda.synchronize();exact(candidate[:n],reference[:n],stage=stage,generation=generation,block=block,warps=warps)
  for block,warps in variants:
   for cold in (False,True):
    ga=graphs['4096_4'];gb=graphs[f'{block}_{warps}'];until=time.monotonic()+.3
    while time.monotonic()<until:sample(ga,cold);sample(gb,cold)
    values={key:[] for key in ('A1','A2','B1','B2')};rng=random.Random(block+warps+int(cold))
    for repeat in range(8):
     arms=list(values);rng.shuffle(arms)
     for arm in arms:values[arm].append(sample(ga if arm.startswith('A') else gb,cold))
    med={key:statistics.median(v) for key,v in values.items()};ratio=lambda x,y:max(x,y)/min(x,y)
    controls={'AA':ratio(med['A1'],med['A2']),'BB':ratio(med['B1'],med['B2'])};drift={key:ratio(statistics.median(v[:4]),statistics.median(v[4:])) for key,v in values.items()}
    r['timings'].append({'stage':stage,'block':block,'warps':warps,'cold_sweep':cold,'samples_us':values,'median_us':med,'controls':controls,'half_drift':drift,'qualified':max(list(controls.values())+list(drift.values()))<=1.1,'speedup_vs_original_spill_control':statistics.median(values['A1']+values['A2'])/statistics.median(values['B1']+values['B2'])});save()
  del graphs,views;gc.collect();torch.cuda.empty_cache()
 r.update(status='COMPLETE_LOCAL_MATERIALIZATION_ONLY',peak_allocated=torch.cuda.max_memory_allocated(),strict_bitwise_gate=all(c['finite'] and c['bitwise'] for c in r['checks']));save()
if __name__=='__main__':main()
