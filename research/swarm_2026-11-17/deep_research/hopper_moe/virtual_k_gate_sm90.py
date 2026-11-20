"""H200-only component numerical gate. No submissions, timing, or serving hooks."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys
P=Path(__file__).resolve().parent
V=P.parents[2]/'swarm_2026-11-16/moe/humming_vendor'
sys.path.insert(0,str(V))
from virtual_k_gate_contract import require_h200, route_arrays

def main():
 raise RuntimeError("HOLD_WRITER_OWNERSHIP_BUG: original virtual epilogue suppresses required writers; use only a future corrected and qualified gate")
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--jobs',nargs='+',default=None)
 p.add_argument('--seeds',type=int,default=3)
 a=p.parse_args()
 if not 1<=a.seeds<=10:raise ValueError('Use 1..10 seeds')
 if a.out.exists():raise ValueError('Fresh output directory required')
 import torch
 # Refuse other devices before weights, launcher import, compilation or kernel launch.
 props=torch.cuda.get_device_properties(torch.cuda.current_device())
 require_h200(props.major,props.minor,props.name)
 if os.environ.get('MAX_JOBS')!='1' or os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')!='1':raise RuntimeError('Set MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1')
 from humming import ops,dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingInputSchema,HummingWeightSchema
 from humming.kernel.humming import HummingKernel
 from virtual_k_codegen import render_virtual_k
 manifest=json.loads((P/'virtual_k_jobs.json').read_text(encoding='utf-8-sig'))
 jobs=[j for j in manifest['jobs'] if j['policy']=='virtual' and (a.jobs is None or j['job_id'] in a.jobs)]
 if not jobs or (a.jobs and set(a.jobs)!={j['job_id'] for j in jobs}):raise ValueError('Unknown or control job')
 for j in jobs:
  if j['configuration']!=j['baseline']:raise ValueError('Baseline must have identical arithmetic and geometry')
 a.out.mkdir(parents=True)
 files=[Path(__file__),P/'virtual_k_gate_contract.py',P/'virtual_k_codegen.py',P/'virtual_k_jobs.json',*sorted(P.glob('virtual_k_*.cuh')),*sorted((V/'humming/include').rglob('*.cuh')),V/'humming/kernel/humming.py',V/'humming/jit/runtime.py',V/'humming/csrc/launcher/launcher.cpp']
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 report={'status':'running','scope':'Direct indexed GEMM component gate; original physical-K baseline versus virtual-K banks. No full-helper, serving or performance claim.', 'gpu':props.name,'sm':[props.major,props.minor],'torch':torch.__version__,'cuda':torch.version.cuda,'source_hashes':{str(f):sha(f) for f in files},'rows':[],'nonfinite_diagnostics':[],'numerical_contract':'All active finite outputs must be bitwise identical, including signed zero. Nonfinite fixtures are diagnostic only: compiler may commute final add operands and NaN payload ordering is unqualified.'}
 def save():
  t=a.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');t.replace(a.out/'results.json')
 def exact(x,y):return bool(torch.equal(x.view(torch.int16),y.view(torch.int16)))
 def make_kernel(c,enabled,dest):
  class IsolatedKernel(HummingKernel):
   def __new__(cls,*args,**kwargs):
    obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
   def prepare(self):
    self.code,self.research_manifest=render_virtual_k(self.code,c,self.sm_version,enabled)
    super().prepare()
  k=IsolatedKernel(**c)
  dest.mkdir();(dest/'kernel.cu').write_text(k.code,encoding='utf-8');(dest/'manifest.json').write_text(json.dumps(k.research_manifest,indent=2),encoding='utf-8')
  return k
 save()
 for job in jobs:
  c=job['configuration'];stage=job['stage'];e=c['num_experts'];n=c['shape_n'];kdim=c['shape_k'];m=job['tokens'];bm=c['block_shape'][0];topk=8 if stage=='w13' else 1;slots=m*8
  dest=a.out/job['job_id'];dest.mkdir()
  baseline=make_kernel(c,False,dest/'baseline');candidate=make_kernel(c,True,dest/'candidate')
  assert baseline.kernel_id!=candidate.kernel_id and baseline.mma_op_class.to_cpp_str()==candidate.mma_op_class.to_cpp_str()
  # Prepare one canonical weight tensor and share it between both launches.
  torch.manual_seed(20261117)
  layer=torch.nn.Module()
  packed=torch.randint(0,256,(e,n,kdim//2),device='cuda',dtype=torch.uint8)
  scales=torch.pow(2.,torch.randint(-8,4,(e,n,kdim//16),device='cuda')).to(torch.float8_e4m3fn)
  layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False))
  layer.register_parameter('weight_scale',torch.nn.Parameter(scales,requires_grad=False))
  schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
  HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=kdim,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,torch_dtype=torch.bfloat16)
  HummingMethod.transform_humming_layer(layer,already_padded=True)
  meta=json.loads(layer.humming_metas[''].to_str())
  for field in ('shape_n','shape_k','num_experts','a_dtype','b_dtype','bs_dtype','c_dtype','weight_scale_group_size','weight_scale_type','weight_scale_2_type','use_packed_k_layout'):
   assert meta[field]==c[field],(field,meta[field],c[field])
  del packed,scales
  weight_ptr=layer.weight.data_ptr();scale_ptr=layer.weight_scale.data_ptr()
  x=torch.zeros((slots//topk,kdim),device='cuda',dtype=torch.bfloat16)
  routes=route_arrays(m,e,bm,'uniform',0)
  sorted_ids=torch.tensor(routes['sorted_ids'],device='cuda',dtype=torch.int32)
  expert_ids=torch.tensor(routes['expert_ids'],device='cuda',dtype=torch.int32)
  padded=torch.tensor([routes['padded']],device='cuda',dtype=torch.int32)
  active=torch.tensor(routes['active'],device='cuda',dtype=torch.bool)
  outputs=[torch.full((slots,n),-123.,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
  tables=[torch.tensor([0,1<<30,ker.kernel_id,0],dtype=torch.int64,device='cpu') for ker in (baseline,candidate)]
  locks=torch.zeros(max(2048,props.multi_processor_count*8),device='cuda',dtype=torch.int32)
  def call(i):
   return ops.launch_kernel(configs=tables[i],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=outputs[i],sorted_ids=sorted_ids,expert_ids=expert_ids,num_tokens_padded=padded,locks=locks,top_k=topk,valid_shape_m=slots)
  for i in range(2):call(i)
  torch.cuda.synchronize()
  graphs=[]
  for i in range(2):
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call(i)
   graphs.append(graph)
  row={'job':job['job_id'],'configuration':c,'baseline_code_sha256':hashlib.sha256(baseline.code.encode()).hexdigest(),'candidate_code_sha256':hashlib.sha256(candidate.code.encode()).hexdigest(),'kernel_ids':[baseline.kernel_id,candidate.kernel_id],'cases':[]}
  report['rows'].append(row);save()
  for seed in range(a.seeds):
   for pattern in ('uniform','hot_expert','mixed_duplicate','all_remote'):
    routes=route_arrays(m,e,bm,pattern,seed)
    sorted_ids.copy_(torch.tensor(routes['sorted_ids'],device='cuda',dtype=torch.int32));expert_ids.copy_(torch.tensor(routes['expert_ids'],device='cuda',dtype=torch.int32));padded.fill_(routes['padded']);active.copy_(torch.tensor(routes['active'],device='cuda',dtype=torch.bool))
    for profile in ('random','cancellation','subnormal','signed_zero'):
     torch.manual_seed(215+seed)
     if profile=='random':x.copy_(torch.randn_like(x)/8)
     elif profile=='cancellation':
      x.copy_(torch.arange(kdim,device='cuda').remainder(2).mul(2).sub(1).to(torch.bfloat16).expand_as(x));x[:,::64]*=32
     elif profile=='subnormal':x.copy_(torch.arange(kdim,device='cuda').remainder(2).mul(2).sub(1).float().mul(2.**-130).to(torch.bfloat16).expand_as(x))
     else:x.zero_();x[:,::2]=-0.
     for output in outputs:output.fill_(-123.)
     call(0);call(1);torch.cuda.synchronize()
     direct=[out.clone() for out in outputs]
     finite=bool(torch.isfinite(direct[0][active]).all() and torch.isfinite(direct[1][active]).all())
     checks={'finite_outputs':finite,'direct_bitwise':exact(direct[0][active],direct[1][active]),'inactive_untouched':all(bool((out[~active]==-123.).all()) for out in direct)}
     for output in outputs:output.fill_(-123.)
     graphs[0].replay();graphs[1].replay();torch.cuda.synchronize()
     checks['changed_input_graph_bitwise']=all(exact(outputs[i],direct[i]) for i in range(2))
     item={'seed':seed,'pattern':pattern,'profile':profile,'active_routes':sum(routes['active']),'checks':checks};row['cases'].append(item);save()
     if not all(checks.values()):raise AssertionError(item)
  # NaN payload and invalid-operation signs are explicitly outside the finite gate.
  routes=route_arrays(m,e,bm,'hot_expert',0)
  sorted_ids.copy_(torch.tensor(routes['sorted_ids'],device='cuda',dtype=torch.int32));expert_ids.copy_(torch.tensor(routes['expert_ids'],device='cuda',dtype=torch.int32));padded.fill_(routes['padded']);active.copy_(torch.tensor(routes['active'],device='cuda',dtype=torch.bool))
  for value in (float('nan'),float('inf')):
   x.fill_(value);call(0);call(1);torch.cuda.synchronize()
   report['nonfinite_diagnostics'].append({'job':job['job_id'],'input':'nan' if value!=value else 'inf','nan_mask_equal':bool(torch.equal(torch.isnan(outputs[0][active]),torch.isnan(outputs[1][active]))),'bitwise_equal_observed':exact(outputs[0][active],outputs[1][active]),'qualification':'Diagnostic only, never an equivalence claim'})
  assert layer.weight.data_ptr()==weight_ptr and layer.weight_scale.data_ptr()==scale_ptr
  del graphs,outputs,direct,x,layer,sorted_ids,expert_ids,padded,active,locks,baseline,candidate
  torch.cuda.synchronize();save()
 report['source_unchanged']=all(sha(Path(f))==v for f,v in report['source_hashes'].items())
 assert report['source_unchanged'];report['status']='finite_component_gate_passed';save()
 print(json.dumps({'status':report['status'],'jobs':len(report['rows']),'finite_cases':sum(len(r['cases']) for r in report['rows'])}),flush=True)
if __name__=='__main__':main()
