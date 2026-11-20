"""Local SM86 exact-layout falsification; actual M32 scored W13 policy."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
from compact_codegen import render_compact
from compact_contract import plan

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);args=pa.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 if (args.out/'results.json').exists():raise ValueError('Fresh receipt required')
 import torch
 props=torch.cuda.get_device_properties();assert (props.major,props.minor)==(8,6)
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
 from compact_routes import CompactRoutes
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 scale=module('compact_scale',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py');split=module('compact_split',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py')
 jobs=json.loads((P/'SM86_COMPACT_JOBS.json').read_text())['jobs'];c0=jobs[0]['configuration'];E,N,K=c0['num_experts'],c0['shape_n'],c0['shape_k'];M=32;S=M*8;cutoff=16;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 files=[P/'compact_routes.py',P/'compact_codegen.py',P/'compact_contract.py',P/'SM86_COMPACT_JOBS.json',Path(__file__),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 report={'status':'running','scope':'Same native arithmetic, actual scored M32 W13 dual-part policy: indexed original-slot versus compact grouped-contiguous. BF16 boundaries and canonical NVFP4 weights unchanged. No H200/serving/performance claim.','gpu':props.name,'source_hashes':{str(f):sha(f) for f in files},'kernels':[],'rows':[]}
 def save():
  t=args.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(args.out/'results.json')
 class Compiler(NVRTCCompiler):
  @classmethod
  def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
 kernels={};tables={};configs={}
 for j in jobs:
  c=j['configuration'];enabled=j['enabled'];label=j['label']
  class Kernel(HummingKernel):
   def __new__(cls,*a,**kw):obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
   @staticmethod
   def _get_compiler():return Compiler
   def prepare(self):
    assert self.sm_version==86 and not self.use_tma and not self.use_warp_spec
    self.code,self.research=render_compact(self.code,c,enabled);super().prepare()
  ker=Kernel(**c);kernels[label]=ker;configs[label]=c;tables[label]=torch.tensor([0,1<<30,ker.kernel_id,0],dtype=torch.int64);d=args.out/label;d.mkdir(exist_ok=True);(d/'kernel.cu').write_text(ker.code);cb=Path(ker.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cb.read_bytes());resource=[]
  for fname in ('stdout.log','stderr.log','cmdline.json'):
   f=cb.parent/fname
   if f.exists():(d/fname).write_bytes(f.read_bytes());resource.extend(line for line in f.read_text(errors='replace').splitlines() if any(s in line for s in ('Used ','spill','stack frame')))
  report['kernels'].append({'label':label,'configuration':c,'resources':resource,'mma':ker.mma_op_class.to_cpp_str(),'cubin_sha256':sha(cb),'research':ker.research});save()
 for policy in ('light','heavy'):assert kernels['indexed_'+policy].mma_op_class.to_cpp_str()==kernels['grouped_'+policy].mma_op_class.to_cpp_str()
 torch.manual_seed(417);layer=torch.nn.Module();packed=torch.randint(0,256,(E,N,K//2),device='cuda',dtype=torch.uint8);sf=torch.pow(2.,torch.randint(-8,4,(E,N,K//16),device='cuda')).to(torch.float8_e4m3fn);layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf,requires_grad=False))
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE);HummingMethod.prepare_layer_meta(layer,shape_n=N,shape_k=K,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=E,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True);del packed,sf
 ids=torch.full((M,8),-1,device='cuda',dtype=torch.int32);x=torch.zeros((M,K),device='cuda',dtype=torch.bfloat16);routes=CompactRoutes(ids,E,cutoff);B0=torch.full((S,N),37.,device='cuda',dtype=torch.bfloat16);B1=torch.full((S,K),37.,device='cuda',dtype=torch.bfloat16);reference=torch.full_like(B0,37.);refact=torch.full_like(B1,37.);scatter_raw=torch.full_like(B0,37.);scatter_act=torch.full_like(B1,37.);scales=torch.linspace(.25,2.,E,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32)
 def run_a(post):
  light,heavy,_=split.partition_routes(ids,E,cutoff)
  for policy,part in [('light',light),('heavy',heavy)]:
   label='indexed_'+policy;align=moe_align_block_size(part,configs[label]['block_shape'][0],E,ignore_invalid_expert=True);ops.launch_kernel(configs=tables[label],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=reference,sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=8,valid_shape_m=S)
  if post:scale.scale_active_rows(reference,ids.flatten(),scales);silu_and_mul(reference,refact,expert_ids=ids.flatten(),expert_step=1)
 def run_b(post):
  routes.update(ids);routes.gather(x,B1)
  for policy,offset in [('light',routes.light),('heavy',routes.heavy)]:ops.launch_kernel(configs=tables['grouped_'+policy],inputs=B1,weight=layer.weight,weight_scale=layer.weight_scale,outputs=B0,expert_layout=offset,locks=locks,top_k=1,valid_shape_m=S)
  if post:scale.scale_active_rows(B0,routes.experts,scales);silu_and_mul(B0,B1,expert_ids=routes.experts,expert_step=1)
 def capture(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 graphs={(mode,side):capture(lambda mode=mode,side=side:(run_a if side=='A' else run_b)(mode=='stage')) for mode in ('raw','stage') for side in ('A','B')}
 def exact(a,b):return bool(torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)))
 for seed in range(3):
  for pattern in ('uniform','onehot','eighthot','last','remote','crossing'):
   rng=random.Random(seed+29);local=[]
   for t in range(M):
    row=rng.sample(range(256),8) if pattern=='uniform' else [0]+rng.sample(range(32,256),7) if pattern=='onehot' else rng.sample(range(8),8) if pattern=='eighthot' else [31]+rng.sample(range(32,256),7) if pattern=='last' else rng.sample(range(32,256),8) if pattern=='remote' else ([seed%2]+rng.sample(range(32,256),7) if t<(15 if seed%2 else 17) else rng.sample(range(32,256),8))
    assert len(set(row))==8;local.extend(v if v<E else -1 for v in row)
   ids.copy_(torch.tensor(local,device='cuda',dtype=torch.int32).view(M,8));torch.manual_seed(seed+83);x.copy_(torch.randn_like(x)/20);active=ids.flatten()>=0;cpu=plan(local,E,cutoff)
   for mode in ('raw','stage'):
    for buf in (B0,B1,reference,refact,scatter_raw,scatter_act):buf.fill_(37.)
    graphs[mode,'A'].replay();graphs[mode,'B'].replay();routes.scatter(B0,scatter_raw)
    if mode=='stage':routes.scatter(B1,scatter_act)
    torch.cuda.synchronize();mapping=routes.map.cpu().tolist();inverse=routes.inverse.cpu().tolist();assert mapping[:cpu['total']]==cpu['compact_to_slot'] and inverse==cpu['slot_to_compact'];assert routes.light.cpu().tolist()==cpu['light_offsets'] and routes.heavy.cpu().tolist()==cpu['heavy_offsets'];checks={'raw_or_scaled_bitwise':exact(reference[active],scatter_raw[active]),'finite':bool(torch.isfinite(reference[active]).all() and torch.isfinite(scatter_raw[active]).all()),'inactive':bool((reference[~active]==37).all() and (scatter_raw[~active]==37).all()),'compact_tail_untouched':bool((B0[cpu['total']:]==37).all())}
    if mode=='stage':checks.update(activation_bitwise=exact(refact[active],scatter_act[active]),activation_inactive=bool((refact[~active]==37).all() and (scatter_act[~active]==37).all()))
    report['rows'].append({'seed':seed,'pattern':pattern,'mode':mode,'active':cpu['total'],'checks':checks});save();assert all(checks.values()),report['rows'][-1]
 # Negative control: unchanged grouped endpoint invents last-expert rows for an empty fixedcapacity allocation.
 ids.fill_(-1);routes.update(ids);B1.zero_();B0.fill_(37.);ops.launch_kernel(configs=tables['grouped_unpatched_negative'],inputs=B1,weight=layer.weight,weight_scale=layer.weight_scale,outputs=B0,expert_layout=routes.light,locks=locks,top_k=1,valid_shape_m=S);torch.cuda.synchronize();report['legacy_endpoint_negative_detected']=bool((B0!=37).any());assert report['legacy_endpoint_negative_detected']
 report['source_unchanged']=all(sha(Path(f))==h for f,h in report['source_hashes'].items());assert report['source_unchanged'];report['status']='same_arithmetic_compact_w13_gate_passed';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save();print(json.dumps({'status':report['status'],'cases':len(report['rows'])}))
if __name__=='__main__':main()
