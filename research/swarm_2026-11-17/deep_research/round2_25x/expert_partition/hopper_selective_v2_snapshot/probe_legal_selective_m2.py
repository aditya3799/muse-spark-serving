"""SM86-only decisive exact K-bank/M-warp reuse experiment; no serving/submission."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,statistics,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
from route_contract import route_arrays
from virtual_k_codegen import render_virtual_k

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',nargs='+',type=int,default=[32]);ap.add_argument('--samples',type=int,default=12);ap.add_argument('--patterns',nargs='+',default=['uniform','hot_one_local','hot_eight_local','mixed','all_remote']);args=ap.parse_args()
 if args.samples<8 or args.tokens!=[32]:raise ValueError('Bounded pilot requires M1..256 and at least8 samples')
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
 jobs=json.loads((P/'LEGAL_M32_JOBS.json').read_text())['jobs'];c0=jobs[0]['configuration'];e,n,kdim=c0['num_experts'],c0['shape_n'],c0['shape_k']
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 files=[P/'configuration_contract.py',P/'selective_routes.py',Path(__file__),P/'LEGAL_M32_JOBS.json',P/'virtual_k_codegen.py',P/'route_contract.py',*sorted(P.glob('*.cuh')),scale_path,*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 report={'status':'running','scope':'SM86 local exact-arithmetic falsification. Actual scored native M32 W13 light/heavy policy versus three partitions. Includes GPU histogram, all alignment and GEMM launches, BF16 scale and SiLU. No H200/serving inference.','gpu':props.name,'SM':[props.major,props.minor],'source_hashes':{str(f):sha(f) for f in files},'kernels':[],'rows':[]}
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
 from selective_routes import partition3
 from configuration_contract import m2_compatible
 split_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py';split_spec=importlib.util.spec_from_file_location('scored_split_reference',split_path);split_module=importlib.util.module_from_spec(split_spec);split_spec.loader.exec_module(split_module)
 report['source_hashes'][str(split_path)]=sha(split_path)
 m=32;slots=m*8;cutoff=16;x=torch.zeros((m,kdim),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,8),device='cuda',dtype=torch.int32)
 outputs={which:torch.full((slots,n),37.,device='cuda',dtype=torch.bfloat16) for which in ('A','B','C')};activations={which:torch.full((slots,n//2),37.,device='cuda',dtype=torch.bfloat16) for which in ('A','B','C')}
 configs={j['label']:j['configuration'] for j in jobs}
 compatible=m2_compatible(configs['heavy'],configs['m2'],cutoff,sm_version=props.major*10+props.minor)
 report['metadata_M2_compatible']=compatible
 def legal_rows(pattern,seed):
  rng=random.Random(seed);global_ids=[]
  for token in range(m):
   if pattern=='uniform':row=rng.sample(range(256),8)
   elif pattern=='hot_one_local':row=[0]+rng.sample(range(32,256),7)
   elif pattern=='hot_eight_local':row=rng.sample(range(8),8)
   elif pattern=='mixed':row=rng.sample(range(4),4)+rng.sample(range(32,256),4) if token<24 else rng.sample(range(256),8)
   elif pattern=='all_remote':row=rng.sample(range(32,256),8)
   else:raise ValueError(pattern)
   assert len(set(row))==8;global_ids.append(row)
  return global_ids,[[value if value<32 else -1 for value in row] for row in global_ids]
 def set_case(pattern,seed):
  global_ids,local=legal_rows(pattern,seed);ids.copy_(torch.tensor(local,device='cuda',dtype=torch.int32));torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20)
  counts=[sum(value==expert for row in local for value in row) for expert in range(32)]
  return ids.flatten()>=0,counts,global_ids
 def launch(label,part,which):
  bm=configs[label]['block_shape'][0];arr=moe_align_block_size(part,bm,e,ignore_invalid_expert=True)
  ops.launch_kernel(configs=tables[label],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=outputs[which],sorted_ids=arr[0],expert_ids=arr[1],num_tokens_padded=arr[2],locks=locks,top_k=8,valid_shape_m=slots)
 def run(which,post=True):
  if which=='A':
   light,heavy,_=split_module.partition_routes(ids,e,cutoff);launch('light',light,which);launch('heavy',heavy,which)
  elif which=='B':
   light,heavy,_=split_module.partition_routes(ids,e,cutoff);launch('light',light,which);launch('m2' if compatible else 'heavy',heavy,which)
  else:
   light,heavy,m2,_=partition3(ids,e,cutoff);launch('light',light,which);launch('heavy',heavy,which);launch('m2',m2,which)
  if post:scale_module.scale_active_rows(outputs[which],ids.flatten(),scales);silu_and_mul(outputs[which],activations[which],expert_ids=ids.flatten(),expert_step=1)
 set_case('uniform',350)
 raw_graphs={which:capture(lambda which=which:run(which,False)) for which in ('A','B','C')};graphs={which:capture(lambda which=which:run(which,True)) for which in ('A','B','C')}
 def check_modes(active):
  for mode,selected in (('raw',raw_graphs),('stage',graphs)):
   for which in ('A','B','C'):outputs[which].fill_(37.);activations[which].fill_(37.);selected[which].replay()
   torch.cuda.synchronize()
   for buffers in ([outputs] if mode=='raw' else [outputs,activations]):
    assert exact(buffers['A'][active],buffers['B'][active]) and exact(buffers['A'][active],buffers['C'][active]),(mode,'bitwise')
    assert bool(torch.isfinite(buffers['A'][active]).all() and torch.isfinite(buffers['B'][active]).all() and torch.isfinite(buffers['C'][active]).all())
    for which in ('A','B','C'):assert bool((buffers[which][~active]==37.).all()),(mode,which,'inactive write')
 for pattern in args.patterns:
  active,counts,global_ids=set_case(pattern,350);check_modes(active)
  light,heavy,m2,gpu_counts=partition3(ids,e,cutoff);masks=[part.flatten()>=0 for part in (light,heavy,m2)];assert torch.equal(gpu_counts,torch.tensor(counts,device='cuda',dtype=torch.int32))
  assert torch.equal(masks[0].int()+masks[1].int()+masks[2].int(),active.int()),'Partition duplication or loss'
  for expert,count in enumerate(counts):
   expected=0 if count<=cutoff else (2 if (count+31)//32<(count+15)//16 else 1)
   for arm,part in enumerate((light,heavy,m2)):
    assert int((part==expert).sum())==(count if arm==expected else 0)
  row={'m':32,'pattern':pattern,'global_top8_unique':all(len(set(v))==8 for v in global_ids),'expert_counts':counts,'active_routes':sum(counts),'raw_scaled_activation_bitwise':True,'inactive_sentinels':True,'GPU_partition_disjoint_complete':True,'timing':{}};report['rows'].append(row);save()
  for changed in ('mixed','all_remote','uniform'):
   changed_active,_,_=set_case(changed,913);scales.mul_(1.03125);check_modes(changed_active)
  row['changed_graph_cases']=6
  active,counts,_=set_case(pattern,350);scales.copy_(torch.linspace(.25,2.,e,device='cuda',dtype=torch.bfloat16))
  base_tiles=sum((c+15)//16 for c in counts);m2_tiles=sum((c+31)//32 for c in counts if c>16);light_tiles=sum((c+15)//16 for c in counts if c<=16)
  row.update(baseline_weight_tiles=base_tiles,candidate_weight_tiles=light_tiles+m2_tiles,baseline_padded_routes=base_tiles*16,candidate_padded_routes=light_tiles*16+m2_tiles*32,byte_model='Requested B/scales tile count; not measured HBM bytes',candidate_GEMM_launches=2,conservative_ablation_GEMM_launches=3,baseline_GEMM_launches=2)
  for variant in ('B','C'):
   for cold in (False,True):
    warmups={}
    for which in ('A',variant):
     warm_start=time.monotonic();iterations=0
     while time.monotonic()-warm_start<.3 or iterations<3:timed(graphs[which],cold);iterations+=1
     warmups[which]={'seconds':time.monotonic()-warm_start,'iterations':iterations}
    measures={key:[] for key in ('A1','B1','A2','B2')}
    for repeat in range(args.samples):
     order=('A1','B1','A2','B2') if repeat%2==0 else ('B2','A2','B1','A1')
     for arm in order:measures[arm].append(timed(graphs[variant if arm.startswith('B') else 'A'],cold))
    med={key:statistics.median(values) for key,values in measures.items()};aa=max(med['A1'],med['A2'])/min(med['A1'],med['A2']);bb=max(med['B1'],med['B2'])/min(med['B1'],med['B2'])
    drift={key:max(statistics.median(values[:len(values)//2]),statistics.median(values[len(values)//2:]))/min(statistics.median(values[:len(values)//2]),statistics.median(values[len(values)//2:])) for key,values in measures.items()}
    row['timing'][variant+('_cold' if cold else '_warm')]={'warmup':warmups,'samples_us':measures,'median_us':med,'AA_ratio':aa,'BB_ratio':bb,'half_drift_ratio':drift,'qualified':aa<=1.10 and bb<=1.10 and max(drift.values())<=1.10,'speedup':statistics.median(measures['A1']+measures['A2'])/statistics.median(measures['B1']+measures['B2'])};save()
  print(json.dumps({'pattern':pattern,'active':sum(counts),'tile_counts':[base_tiles,light_tiles+m2_tiles],'timing':{key:{'speedup':value['speedup'],'qualified':value['qualified']} for key,value in row['timing'].items()}}),flush=True)
 report['source_unchanged']=all(sha(Path(f))==v for f,v in report['source_hashes'].items());assert report['source_unchanged'];report['status']='completed';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
if __name__=='__main__':main()
