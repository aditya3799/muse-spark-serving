"""SM86-only decisive exact K-bank/M-warp reuse experiment; no serving/submission."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,statistics,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
from route_contract import route_arrays
from virtual_k_codegen import render_virtual_k

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',nargs='+',type=int,default=[8,128]);ap.add_argument('--samples',type=int,default=12);ap.add_argument('--patterns',nargs='+',default=['uniform','hot_expert']);args=ap.parse_args()
 if args.samples<8 or any(m<=0 or m>256 for m in args.tokens):raise ValueError('Bounded pilot requires M1..256 and at least8 samples')
 args.out.mkdir(parents=True,exist_ok=True)
 if (args.out/'results.json').exists():raise ValueError('Fresh receipt required')
 import torch
 props=torch.cuda.get_device_properties();assert (props.major,props.minor)==(8,6),'SM86-only falsification; no Hopper inference'
 assert os.environ.get('MAX_JOBS')=='1' and os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')=='1'
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 from humming import ops,dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingWeightSchema,HummingInputSchema
 from humming.kernel.humming import HummingKernel
 from humming.jit.compiler import NVRTCCompiler
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 from sglang.kernels.ops.activation.activation import silu_and_mul
 scale_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py';spec=importlib.util.spec_from_file_location('reuse_scale',scale_path);scale_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scale_module;spec.loader.exec_module(scale_module)
 jobs=json.loads((P/'SM86_REUSE_JOBS.json').read_text())['jobs'];c0=jobs[0]['configuration'];e,n,kdim=c0['num_experts'],c0['shape_n'],c0['shape_k']
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 files=[Path(__file__),P/'SM86_REUSE_JOBS.json',P/'virtual_k_codegen.py',P/'route_contract.py',*sorted(P.glob('*.cuh')),scale_path,*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 report={'status':'running','scope':'SM86 local exact-arithmetic falsification. Fixed W13 configuration; compare kernel-only and GPU alignment+GEMM+BF16 scale+SiLU. No H200/serving inference.','gpu':props.name,'SM':[props.major,props.minor],'source_hashes':{str(f):sha(f) for f in files},'kernels':[],'rows':[]}
 def save():
  t=args.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');t.replace(args.out/'results.json')
 class Compiler(NVRTCCompiler):
  @classmethod
  def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
 kernels={};tables={}
 for job in jobs:
  config=job['configuration'];enabled=job['enabled'];label=job['label']
  class ResearchKernel(HummingKernel):
   def __new__(cls,*a,**kw):
    obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
   @staticmethod
   def _get_compiler():return Compiler
   def prepare(self):
    assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec
    self.code,self.research=render_virtual_k(self.code,config,86,enabled)
    super().prepare()
  kernel=ResearchKernel(**config);kernels[label]=kernel;tables[label]=torch.tensor([0,1<<30,kernel.kernel_id,0],device='cpu',dtype=torch.int64)
  d=args.out/label;d.mkdir(exist_ok=True);(d/'kernel.cu').write_text(kernel.code,encoding='utf-8')
  cubin=Path(kernel.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cubin.read_bytes());resources=[]
  for name in ('stdout.log','stderr.log','cmdline.json'):
   f=cubin.parent/name
   if f.exists():
    (d/name).write_bytes(f.read_bytes());resources.extend(line for line in f.read_text(errors='replace').splitlines() if any(x in line for x in ('Used ','spill','stack frame')))
  item={'label':label,'configuration':config,'research':kernel.research,'shared_estimated':kernel.estimated_smem_size,'native_mma':kernel.mma_op_class.to_cpp_str(),'native_mma_sha256':hashlib.sha256(kernel.mma_op_class.to_cpp_str().encode()).hexdigest(),'resources':resources,'cubin_sha256':sha(cubin)};report['kernels'].append(item);save()
 assert len(set(r['native_mma_sha256'] for r in report['kernels']))==1,'Native MMA expression changed'
 torch.manual_seed(219);layer=torch.nn.Module();packed=torch.randint(0,256,(e,n,kdim//2),device='cuda',dtype=torch.uint8);sf=torch.pow(2.,torch.randint(-8,4,(e,n,kdim//16),device='cuda')).to(torch.float8_e4m3fn)
 layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf,requires_grad=False))
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
 HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=kdim,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True);del packed,sf
 scales=torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32);flush=torch.empty((64*1024**2,),device='cuda',dtype=torch.uint8)
 def exact(a,b):return bool(torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)))
 def capture(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 def timed(graph,cold):
  if cold:flush.fill_(71)
  start.record();graph.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000
 for m in args.tokens:
  x=torch.zeros((m,kdim),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,8),device='cuda',dtype=torch.int32);slots=m*8
  outputs={j['label']:torch.full((slots,n),37.,device='cuda',dtype=torch.bfloat16) for j in jobs};activations={j['label']:torch.full((slots,n//2),37.,device='cuda',dtype=torch.bfloat16) for j in jobs}
  aligned={}
  for job in jobs:
   r=route_arrays(m,e,job['configuration']['block_shape'][0],'uniform',0);aligned[job['label']]=[torch.tensor(r['sorted_ids'],device='cuda',dtype=torch.int32),torch.tensor(r['expert_ids'],device='cuda',dtype=torch.int32),torch.tensor([r['padded']],device='cuda',dtype=torch.int32)]
  def set_case(pattern,seed):
   for job in jobs:
    r=route_arrays(m,e,job['configuration']['block_shape'][0],pattern,seed);arr=aligned[job['label']];arr[0].copy_(torch.tensor(r['sorted_ids'],device='cuda',dtype=torch.int32));arr[1].copy_(torch.tensor(r['expert_ids'],device='cuda',dtype=torch.int32));arr[2].fill_(r['padded'])
   ids.copy_(torch.tensor(r['local_ids'],device='cuda',dtype=torch.int32).view(m,8));torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20)
   return ids.flatten()>=0,r
  def launch(label,arr):return ops.launch_kernel(configs=tables[label],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=outputs[label],sorted_ids=arr[0],expert_ids=arr[1],num_tokens_padded=arr[2],locks=locks,top_k=8,valid_shape_m=slots)
  def stage(label,bm):
   arr=moe_align_block_size(ids,bm,e,ignore_invalid_expert=True);launch(label,arr);scale_module.scale_active_rows(outputs[label],ids.flatten(),scales);silu_and_mul(outputs[label],activations[label],expert_ids=ids.flatten(),expert_step=1)
  set_case('uniform',318+m)
  graphs={'kernel':{j['label']:capture(lambda j=j:launch(j['label'],aligned[j['label']])) for j in jobs},'stage':{j['label']:capture(lambda j=j:stage(j['label'],j['configuration']['block_shape'][0])) for j in jobs}}
  for pattern in args.patterns:
   active,r=set_case(pattern,318+m);counts=[r['local_ids'].count(i) for i in range(e)]
   row={'m':m,'pattern':pattern,'active_routes':sum(counts),'expert_counts':counts,'variants':[]};report['rows'].append(row);save()
   for mode in ('kernel','stage'):
    for label in outputs:outputs[label].fill_(37.);activations[label].fill_(37.);graphs[mode][label].replay()
    torch.cuda.synchronize();values=outputs if mode=='kernel' else activations
    ref=values['baseline'][active].clone();assert bool(torch.isfinite(ref).all())
    for label in values:
     assert exact(values[label][active],ref),(m,pattern,mode,label,'numerical mismatch')
     assert bool((values[label][~active]==37.).all()),(m,pattern,mode,label,'inactive write')
   for changed in ('hot_expert','uniform','all_remote'):
    changed_active,_=set_case(changed,913+m);scales.mul_(1.03125)
    for mode in ('kernel','stage'):
     for label in outputs:outputs[label].fill_(37.);activations[label].fill_(37.);graphs[mode][label].replay()
     torch.cuda.synchronize();values=outputs if mode=='kernel' else activations
     for label in values:
      assert exact(values[label][changed_active],values['baseline'][changed_active]),(m,changed,mode,label,'changed graph mismatch')
      assert bool((values[label][~changed_active]==37.).all()),(m,changed,mode,label,'changed inactive write')
   active,r=set_case(pattern,318+m);scales.copy_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16))
   for job in jobs[1:]:
    label=job['label'];bm=job['configuration']['block_shape'][0];base_bm=c0['block_shape'][0];tiles=sum((count+bm-1)//bm for count in counts);base_tiles=sum((count+base_bm-1)//base_bm for count in counts)
    item={'label':label,'raw_and_stage_bitwise':True,'changed_graph_cases':6,'padded_routes':tiles*bm,'baseline_padded_routes':base_tiles*base_bm,'scheduled_B_scale_bytes':tiles*n*kdim*(.5+1/16),'baseline_scheduled_B_scale_bytes':base_tiles*n*kdim*(.5+1/16),'unique_B_scale_bytes':sum(v>0 for v in counts)*n*kdim*(.5+1/16),'byte_model_limit':'Requested packed B/scales across CTA tiles, not measured HBM bytes; L2 reuse may reduce device traffic.','timing':{}};row['variants'].append(item);save()
    for mode in ('kernel','stage'):
     for cold in (False,True):
      measures={'A1':[],'B1':[],'B2':[],'A2':[]}
      for repeat in range(args.samples):
       order=('A1','B1','A2','B2') if repeat%2==0 else ('B2','A2','B1','A1')
       for arm in order:measures[arm].append(timed(graphs[mode][label if arm.startswith('B') else 'baseline'],cold))
      med={k:statistics.median(v) for k,v in measures.items()};aa=max(med['A1'],med['A2'])/min(med['A1'],med['A2']);item['timing'][mode+('_cold' if cold else '_warm')]={'samples_us':measures,'median_us':med,'AA_ratio':aa,'AA_pass':aa<=1.10,'BB_ratio':max(med['B1'],med['B2'])/min(med['B1'],med['B2']),'BB_pass':max(med['B1'],med['B2'])/min(med['B1'],med['B2'])<=1.10,'speedup':statistics.median(measures['A1']+measures['A2'])/statistics.median(measures['B1']+measures['B2'])};save()
   print(json.dumps({'m':m,'pattern':pattern,'active':sum(counts),'variants':[(v['label'],{k:t['speedup'] for k,t in v['timing'].items()}) for v in row['variants']]}),flush=True)
  del graphs,outputs,activations,aligned,x,ids;torch.cuda.synchronize()
 report['source_unchanged']=all(sha(Path(f))==v for f,v in report['source_hashes'].items());assert report['source_unchanged'];report['status']='completed';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
if __name__=='__main__':main()
