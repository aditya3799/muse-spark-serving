"""Exact exported v3 finalizer correctness only; no timing inference."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import torch
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--sglang-root',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
paths={
 'base':(a.sglang_root/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py','d87fcf2d67a6ae92d3b2e08d9dfa670dbdb7663c4683a6cb933d5c3b8cf42fe8'),
 'fused':(P/'candidate_v3/serving_export/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py','1244e7e0e49c878d51b9ce022e465fe87a105ccdac3275411c9685d865e2a3a4'),
 'scale':(P/'candidate_v2/serving_export/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py','1fb726a24a8a3335f6b894b602223ca252783607594aea0d108e3c01637bbddd'),
}
mods={}
for name,(path,digest) in paths.items():
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 spec=importlib.util.spec_from_file_location('v3_probe_'+name,path);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);mods[name]=mod
base=mods['base'].fused_post_expert_norm_weighted_sum;fused=mods['fused'].fused_post_expert_norm_weighted_sum;scale=mods['scale'].scale_active_rows
report={'status':'running','scope':'Exact exported finalizer GPU correctness; compare full final output bits against separate BF16 scaling + unchanged finalizer. This does not independently observe intermediate rounding bits or promise NaN payload preservation; no timing measurements','harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sources':{str(p):s for p,s in paths.values()},'gpu':torch.cuda.get_device_name(),'rows':[],'edge_rows':[]}
def save():(a.out/'results.json').write_text(json.dumps(report,indent=2))
def equal(actual,expected,label):
 result=torch.equal(actual.view(torch.int16),expected.view(torch.int16))
 if not result:report.update(status='failed',failure=label);save();raise AssertionError(label)
 return True
def capture(call):
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):call()
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=call()
 return graph,out
safe_scales=[0.,-0.,1.,-1.,.1,1.5,2.**-133,2.**-126,2.**-7,2.**7,2.**40,-2.**40]
for m,h,id_dtype in [(m,4096,torch.int32) for m in (1,2,8,9,32)]+[(9,3072,torch.int64)]:
 for mode in ('weighted_first','weighted_after','unweighted','identity','round_product'):
  torch.manual_seed(730+m);topk=8
  x=torch.randn((m,topk,h),device='cuda',dtype=torch.bfloat16)/20
  ids=((torch.arange(m*topk,device='cuda',dtype=id_dtype)*5)%len(safe_scales)).view(m,topk);ids[:,1]=ids[:,0];ids[::3,-1]=-1
  original_ids=ids.clone();scales=torch.tensor(safe_scales,device='cuda',dtype=torch.bfloat16)
  scores=torch.rand((m,topk),device='cuda');scores/=scores.sum(-1,keepdim=True)
  norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16)
  opts={'norm_weight':norm if mode.startswith('weighted') else None,'eps':1e-8,'weight_first':mode!='weighted_after','apply_norm':mode not in ('identity','round_product'),'round_product':mode=='round_product','slot_expert_ids':ids}
  def reference():
   copied=x.clone();scale(copied.view(m*topk,h),ids,scales);return base(copied,scores,**opts)
  def candidate():return fused(x,scores,expert_output_scale=scales,**opts)
  equal(candidate(),reference(),[m,mode,'eager']);gb,ob=capture(reference);gc,oc=capture(candidate)
  checks=[]
  for step in range(3):
   torch.manual_seed(830+step);x.copy_(torch.randn_like(x)/20);scores.copy_(torch.rand_like(scores));scores/=scores.sum(-1,keepdim=True)
   ids.copy_(original_ids if step!=1 else torch.full_like(ids,-1))
   scales.copy_(torch.tensor(safe_scales if step==2 else list(reversed(safe_scales)),device='cuda',dtype=torch.bfloat16));norm.copy_(torch.linspace(.25+step/4,1.25+step/4,h,device='cuda',dtype=torch.bfloat16))
   gb.replay();gc.replay();torch.cuda.synchronize();equal(oc,ob,[m,mode,step]);assert torch.isfinite(oc).all()
   if step==1:assert (oc==0).all()
   checks.append({'step':step,'bitwise_equal':True,'all_remote':step==1})
  report['rows'].append({'m':m,'mode':mode,'hidden':h,'topk':topk,'id_dtype':str(id_dtype),'eager_bitwise_equal':True,'graph':checks});save();print(json.dumps({'m':m,'mode':mode,'passed':True}),flush=True)
  del gb,gc,ob,oc;torch.cuda.synchronize()
  if (a.out/'STOP_AFTER_CASE').exists():report['status']='stopped_cooperatively';save();sys.exit(0)
# Exhaustive finite BF16 bit patterns, including signed zeros and subnormals.
values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16);values=values[torch.isfinite(values)].cuda().reshape(64,1,1020)
ids=torch.zeros((64,1),device='cuda',dtype=torch.int32);scores=torch.ones((64,1),device='cuda');scales=torch.ones((1,),device='cuda',dtype=torch.bfloat16)
for value in [0.,-0.,1.,-1.,.1,-.1,.5,1.5,2.**-133,2.**-126,2.**-7,2.**7,2.**126,-2.**126]:
 scales.fill_(value);copied=values.clone();scale(copied.view(64,1020),ids,scales)
 expected=base(copied,scores,apply_norm=False,slot_expert_ids=ids)
 actual=fused(values,scores,apply_norm=False,slot_expert_ids=ids,expert_output_scale=scales)
 equal(actual,expected,['edge',value]);report['edge_rows'].append({'scale_after_bf16':float(scales.cpu()[0]),'finite_patterns':65280,'bitwise_equal':True});save()
empty=torch.empty((0,8,4096),device='cuda',dtype=torch.bfloat16)
assert fused(empty,torch.empty((0,8),device='cuda'),slot_expert_ids=torch.empty((0,8),device='cuda',dtype=torch.int32),expert_output_scale=scales).shape==(0,4096)
report.update(status='completed',passed=True,zero_rows_passed=True);save();torch.cuda.synchronize();print('completed',flush=True)
