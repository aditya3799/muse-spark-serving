"""Single-GPU exact row-norm arithmetic and component timing; no collective claim."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,math,statistics,sys,time
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',type=int,nargs='+',default=[512,1024,4096,16384]);p.add_argument('--samples',type=int,default=12);a=p.parse_args()
import os
if os.environ.get('ROW_NORM_BOUND')!='240':raise RuntimeError('Use supervising bounded launcher')
if a.samples<8 or a.samples%2:raise ValueError('Even samples>=8 required')
if any(m<512 or m%8 for m in a.tokens):raise ValueError('Positive supported TP8 row sizes required')
a.out.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text());base=P.parents[2]/'coordination/round5_humming_adaptive_fused/candidate_sources'
helper_rel='python/sglang/srt/layers/communication/muse_spark_gemm_ag.py';candidate=P/'candidate_sources'/helper_rel;baseline=base/helper_rel
for name,values in manifest['sources'].items():
 assert sha(P/'candidate_sources'/name)==values['candidate'];assert sha(base/name)==values['baseline']
import torch
import torch.nn.functional as F
from torch import nn
import enum,math
norm_path=a.sglang_root/'python/sglang/srt/layers/muse_spark/muse_spark_layers.py'
assert sha(norm_path)=='930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0'
tree=ast.parse(norm_path.read_text());norm_nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('GainCenterType','calculate_gain_center','MuseSpark1xRMSNorm')]
ns={'torch':torch,'nn':nn,'F':F,'enum':enum,'math':math,'Optional':__import__('typing').Optional,'_is_cpu':False};exec(compile(ast.Module(body=norm_nodes,type_ignores=[]),str(norm_path),'exec'),ns);Norm=ns['MuseSpark1xRMSNorm']
def helper(path):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prefill_row_projection');d={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),d);return d[node.name],ast.unparse(node)
baseline_fn,baseline_text=helper(baseline);candidate_fn,candidate_text=helper(candidate)
(a.out/'baseline_executed_helper.py').write_text(baseline_text);(a.out/'candidate_executed_helper.py').write_text(candidate_text)
class LocalRowsOnly:
 """Return local tensor only: explicitly no all-gather emulation or protocol test."""
 world_size=8
 def __init__(self,rank):self.rank_in_group=rank;self.calls=0
 def all_gather(self,local,dim=0):assert dim==0;self.calls+=1;return local
assert torch.cuda.is_available();torch.set_grad_enabled(False);torch.manual_seed(216)
report={'status':'running','argv':sys.argv,'harness_sha256':sha(Path(__file__)),'manifest_sha256':sha(P/'manifest.json'),'norm_source_sha256':sha(norm_path),'helper_hashes':{'baseline':sha(baseline),'candidate':sha(candidate)},'scope':'RTX3050 single-GPU arithmetic of exact full-feature row normalization and exact local projection helpers. all_gather callback returns local tensor; concatenation by test forms reference. No NCCL, multicast, actual communication, multiGPU graph, whole-model memory or serving proof. Timing measures normalization component only.','torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'geometry':{'K':8192,'N':4096,'simulated_row_partitions':8,'tokens':a.tokens},'projection_rows':[],'norm_rows':[],'timings':[],'memory':[],'gate':{'relative_l2':.003,'max_abs':.03125}}
def save():(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def compare(value,reference,label):
 diff=value.float()-reference.float();rel=(diff.norm()/reference.float().norm().clamp_min(1e-20)).item();absolute=diff.abs().max().item();bits=bool(torch.equal(value.view(torch.int16),reference.view(torch.int16)));passed=bool(torch.isfinite(value).all()) and rel<=.003 and absolute<=.03125
 item={'label':label,'relative_l2':rel,'max_abs':absolute,'bitwise_equal':bits,'passed':passed}
 if not passed:report.update(status='numerical_failure',failure=item);save();raise AssertionError(item)
 return item
def capture(fn):
 for _ in range(2):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=fn()
 return g,out
print(json.dumps({'milestone':'imports_ready','gpu':report['gpu']}),flush=True)
weight=torch.randn((4096,8192),device='cuda',dtype=torch.bfloat16)/32
norm=Norm(4096,eps=1e-5,has_weight=True,zero_centered_gamma=True,residual_id=2).to(device='cuda',dtype=torch.bfloat16);norm.weight.copy_(torch.linspace(-.5,.5,4096,device='cuda',dtype=torch.bfloat16));norm.precompute_effective_weight();flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
print(json.dumps({'milestone':'weights_ready'}),flush=True);save()
for m in a.tokens:
 x=torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)/20
 raw=[];normalized=[]
 for rank in range(8):
  group=LocalRowsOnly(rank);raw.append(baseline_fn(group,x,weight));normalized.append(candidate_fn(group,x,weight,row_norm=norm));assert group.calls==2
 full=torch.cat(raw,dim=0);joined=torch.cat(normalized,dim=0);reference=norm(full);torch.cuda.synchronize();report['projection_rows'].append({'m':m,**compare(joined,reference,'exact_local_helpers_then_row_order')});del raw,normalized,joined,reference,x
 # Fixed-size GPU buffers for changed-input graph replay: B reads rank7 row view.
 local=full[(7*m)//8:]
 for eps in (1e-6,1e-3,1e-5):
  # Epsilon is a scalar capture parameter: each epsilon receives fresh graphs.
  norm.eps=eps
  ga,oa=capture(lambda:norm(full));gb,ob=capture(lambda:norm(local))
  for seed,pattern in ((221,'gaussian'),(227,'large'),(239,'zeros'),(251,'small')):
   torch.manual_seed(seed)
   if pattern=='zeros':full.zero_()
   else:full.copy_(torch.randn_like(full)*(100 if pattern=='large' else (1e-5 if pattern=='small' else .05)))
   # Change the effective weight contents at its stable captured address.
   norm._frozen_weight.copy_(torch.linspace(-.5,1.5,4096,device='cuda',dtype=torch.bfloat16)*(0 if pattern=='zeros' else (2 if pattern=='large' else 1)))
   ga.replay();gb.replay();torch.cuda.synchronize();report['norm_rows'].append({'m':m,'pattern':pattern,'seed':seed,'eps':eps,'changed_effective_weight':True,**compare(ob,oa[(7*m)//8:],'changed_input_and_weight_graph_rank7')})
  if eps!=1e-5:del ga,gb,oa,ob;torch.cuda.synchronize()
 # Restore representative Gaussian input and effective weights before timing.
 norm._frozen_weight.copy_((norm.weight+norm.gain_center).detach())
 torch.manual_seed(251);full.copy_(torch.randn_like(full)*.05)
 def sample(g):
  flush.zero_();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();g.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 labels=['A1','B1','A2','B2'];samples={label:[] for label in labels}
 for order in (labels,labels[::-1]):
  for label in order:sample(ga if label[0]=='A' else gb)
 for rep in range(a.samples):
  for label in (labels if rep%2==0 else labels[::-1]):samples[label].append(sample(ga if label[0]=='A' else gb))
 med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'AB'};controls={k:med[k+'1']/med[k+'2'] for k in 'AB'}
 report['timings'].append({'m':m,'samples_us':samples,'label_medians_us':med,'arm_medians_us':arms,'component_speedup':arms['A']/arms['B'],'saved_component_us':arms['A']-arms['B'],'controls':controls,'noise_pass':all(abs(v-1)<=.1 for v in controls.values()),'scope':'Full-row-count norm versus rank-local norm, same exact class/weights; excludes GEMM and communication'})
 del ga,gb,oa,ob;torch.cuda.synchronize()
 for label,tensor in (('full',full),('rank_local',local)):
  torch.cuda.reset_peak_memory_stats();start_alloc=torch.cuda.memory_allocated();out=norm(tensor);torch.cuda.synchronize();report['memory'].append({'m':m,'scope':label,'allocated_peak_extra_bytes':torch.cuda.max_memory_allocated()-start_alloc,'output_bytes':out.numel()*out.element_size()});del out
 del full,local;torch.cuda.synchronize();save();print(json.dumps({'milestone':'shape_complete','m':m,'timing':report['timings'][-1]['component_speedup'],'noise_pass':report['timings'][-1]['noise_pass']}),flush=True)
report.update(status='completed',passed=True,bitwise_equal_all=all(x['bitwise_equal'] for x in report['projection_rows']+report['norm_rows']),timing_noise_failures=sum(not x['noise_pass'] for x in report['timings']));save();print(json.dumps({'status':'completed','bitwise_equal_all':report['bitwise_equal_all']}),flush=True)