"""Exact production split boundary/dispatch proof; no timing measurements."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(P/'humming_vendor'));sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--sglang-root',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
from kernel_runtime import bootstrap
runtime=bootstrap(a.sglang_root)
import torch
path=P.parent/'projection_communication/adaptive_split_v3/serving_export/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest=='752038f214e7f8e21e16c4af8de2c9bb43a97733e7bf61615ddbe125124d5bc8'
name='sglang.kernels.ops.moe.muse_spark_humming_split';spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
scale_path=path.with_name('muse_spark_humming_scale.py');scale_name='sglang.kernels.ops.moe.muse_spark_humming_scale';spec=importlib.util.spec_from_file_location(scale_name,scale_path);scale_mod=importlib.util.module_from_spec(spec);sys.modules[scale_name]=scale_mod;spec.loader.exec_module(scale_mod)
report={'status':'running','scope':'Exact production partition kernel at4096routes plus actual production fallback dispatch with spy; no MoE arithmetic or performance claim for spy calls','source_sha256':digest,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime':runtime,'gpu':torch.cuda.get_device_name(),'rows':[],'fallback_rows':[]}
def save():(a.out/'results.json').write_text(json.dumps(report,indent=2))
patterns={}
patterns['light']=[i%32 for i in range(128)]+[-1]*(4096-128)
patterns['heavy']=[i%32 for i in range(4096)]
patterns['threshold']=[0]*15+[1]*16+[2]*17+[-1]*(4096-48)
patterns['mixed']=[0]*128+[1]*15+[2]*16+[3]*17+[-1]*(4096-176)
patterns['all_remote']=[-1]*4096
for dtype in (torch.int32,torch.int64):
 ids=torch.tensor(patterns['light'],device='cuda',dtype=dtype).reshape(512,8)
 for _ in range(3):mod.partition_routes(ids,32,16)
 torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):light,heavy,counts=mod.partition_routes(ids,32,16)
 for pattern in ('light','heavy','threshold','mixed','all_remote','light'):
  ids.copy_(torch.tensor(patterns[pattern],device='cuda',dtype=dtype).reshape(512,8));graph.replay();torch.cuda.synchronize()
  valid=ids>=0;expected_counts=torch.bincount(ids[valid].long(),minlength=32).to(torch.int32);mapped_counts=expected_counts[ids.clamp_min(0).long()]
  expected_light=torch.where(valid&(mapped_counts<=16),ids,-1).to(torch.int32);expected_heavy=torch.where(valid&(mapped_counts>16),ids,-1).to(torch.int32)
  assert torch.equal(counts,expected_counts) and torch.equal(light,expected_light) and torch.equal(heavy,expected_heavy)
  assert not bool(((light>=0)&(heavy>=0)).any());assert torch.equal((light>=0)|(heavy>=0),valid)
  report['rows'].append({'dtype':str(dtype),'pattern':pattern,'routes':4096,'experts':32,'cutoff':16,'passed':True});save()
 del graph,light,heavy,counts;torch.cuda.synchronize()
# Actual exported split dispatcher must return the supplied default callback above limit.
layer=SimpleNamespace(w13_post_scale=torch.ones((32,),device='cuda',dtype=torch.bfloat16),w2_post_scale=torch.ones((32,),device='cuda',dtype=torch.bfloat16))
x=torch.zeros((513,64),device='cuda',dtype=torch.bfloat16);ids=torch.zeros((513,8),device='cuda',dtype=torch.int32)
for retain in (False,True):
 for defer in (False,True):
  marker=object();seen=[]
  def fallback(received_layer,received_x,received_ids,received_torch,*,retain=False,defer_downscale=False):
   assert received_layer is layer and received_x is x and received_ids is ids and received_torch is torch
   seen.append((retain,defer_downscale));return marker
  assert mod.split_expert_outputs(fallback,layer,x,ids,torch,256,retain=retain,defer_downscale=defer) is marker
  assert seen==[(retain,defer)]
  report['fallback_rows'].append({'routes':4104,'retain':retain,'defer_downscale':defer,'callback_once':True,'scope':'dispatch only; no arithmetic'})
try:mod.partition_routes(ids,32,16)
except ValueError:report['partition_limit_rejected']=True
else:raise AssertionError('Partition kernel should reject oversized direct call')
report.update(status='completed',passed=True);save();torch.cuda.synchronize();print(json.dumps({'passed':True,'replays':len(report['rows']),'fallback_cases':len(report['fallback_rows'])}),flush=True)
