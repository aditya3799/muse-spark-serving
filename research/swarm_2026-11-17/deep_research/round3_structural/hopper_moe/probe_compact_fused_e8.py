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
 free_before,total_memory=torch.cuda.mem_get_info();reserve=512*1024**2;estimated_peak=1024*1024**2
 if free_before<reserve+estimated_peak:raise RuntimeError(f'Insufficient shared-GPU headroom: free={free_before}, required={reserve+estimated_peak}; TTS remains untouched')
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
 from compact_fused_runtime import CompactFusedKernel,admit_both
 def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 scale=module('compact_scale',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py');split=module('compact_split',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py')
 jobs=json.loads((P/'SM86_COMPACT_E8_FUSED_JOBS.json').read_text())['jobs'];c0=jobs[0]['configuration'];E,N,K=c0['num_experts'],c0['shape_n'],c0['shape_k'];M=32;S=M*8;cutoff=16;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 files=[P/'compact_routes.py',P/'compact_codegen.py',P/'compact_contract.py',P/'SM86_COMPACT_E8_FUSED_JOBS.json',Path(__file__),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 files.extend([P/'compact_fused_codegen.py',P/'compact_fused_runtime.py',P.parents[1]/'coordination/w13_fused_epilogue.py',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_split.py',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py',*sorted((OLD/'humming_vendor/humming').rglob('*.py'))]);import inspect;files.append(Path(inspect.getfile(silu_and_mul)))
 report={'status':'running','scope':'Compact fusedW13 two-buffer E8 M32 same tile/K arithmetic, one interleaved candidate representation and exact scoredW2/finalizer: indexed original-slot versus compact grouped-contiguous. BF16 boundaries and canonical NVFP4 weights unchanged. No H200/serving/performance claim.','gpu':props.name,'fixture_experts':8,'memory_before_bytes':free_before,'memory_reserve_bytes':reserve,'peak_budget_bytes':estimated_peak,'shared_gpu_timing':'Any later timing is provisional while independent TTS server remains active','source_hashes':{str(f):sha(f) for f in files},'kernels':[],'rows':[],'scratch_contract':'B0 gatheredX then compactdown; B1 fused activation then original-slot down. Both light and heavy W13 fused before admission. Scatter skips inactive slots, which finalizer ignores. No inactive sentinel claim after alias reuse.'}
 def save():
  t=args.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(args.out/'results.json')
 class Compiler(NVRTCCompiler):
  @classmethod
  def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
 admit_both({policy:next(j['configuration'] for j in jobs if j['label']=='grouped_'+policy) for policy in ('light','heavy')})
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
  ker=CompactFusedKernel(**c) if j.get('fused') else Kernel(**c);kernels[label]=ker;configs[label]=c
  if not j.get('fused'):tables[label]=torch.tensor([0,1<<30,ker.kernel_id,0],dtype=torch.int64)
  d=args.out/label;d.mkdir(exist_ok=True);(d/'kernel.cu').write_text(ker.code);cb=Path(ker.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cb.read_bytes());resource=[]
  for fname in ('stdout.log','stderr.log','cmdline.json'):
   f=cb.parent/fname
   if f.exists():(d/fname).write_bytes(f.read_bytes());resource.extend(line for line in f.read_text(errors='replace').splitlines() if any(s in line for s in ('Used ','spill','stack frame')))
  report['kernels'].append({'label':label,'configuration':c,'resources':resource,'mma':ker.mma_op_class.to_cpp_str(),'cubin_sha256':sha(cb),'research':ker.research});save()
 for prefix in ('','w2_'):
  for policy in ('light','heavy'):assert kernels[prefix+'indexed_'+policy].mma_op_class.to_cpp_str()==kernels[prefix+'grouped_'+policy].mma_op_class.to_cpp_str()
 torch.manual_seed(417);layer=torch.nn.Module();packed=torch.randint(0,256,(E,N,K//2),device='cuda',dtype=torch.uint8);sf=torch.pow(2.,torch.randint(-8,4,(E,N,K//16),device='cuda')).to(torch.float8_e4m3fn);layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf,requires_grad=False))
 perm=torch.stack((torch.arange(K,device='cuda'),torch.arange(K,2*K,device='cuda')),dim=1).flatten();candidate=torch.nn.Module();candidate.register_parameter('weight',torch.nn.Parameter(packed[:,perm].contiguous().view(torch.int32),requires_grad=False));candidate.register_parameter('weight_scale',torch.nn.Parameter(sf.view(torch.uint8)[:,perm].contiguous().view(torch.float8_e4m3fn),requires_grad=False));candidate.research_weight_layout='gate_up_interleaved'
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE);HummingMethod.prepare_layer_meta(layer,shape_n=N,shape_k=K,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=E,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True);HummingMethod.prepare_layer_meta(candidate,shape_n=N,shape_k=K,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=E,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(candidate,already_padded=True);del packed,sf

 down_layer=torch.nn.Module();packed2=torch.randint(0,256,(E,K,K//2),device='cuda',dtype=torch.uint8);sf2=torch.pow(2.,torch.randint(-8,4,(E,K,K//16),device='cuda')).to(torch.float8_e4m3fn);down_layer.register_parameter('weight',torch.nn.Parameter(packed2.view(torch.int32),requires_grad=False));down_layer.register_parameter('weight_scale',torch.nn.Parameter(sf2,requires_grad=False));HummingMethod.prepare_layer_meta(down_layer,shape_n=K,shape_k=K,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=E,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(down_layer,already_padded=True);del packed2,sf2
 finalizer=module('compact_finalizer',EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py');norm=torch.linspace(.5,1.5,K,device='cuda',dtype=torch.bfloat16);down_scales=torch.linspace(.5,1.25,E,device='cuda',dtype=torch.bfloat16);scores=torch.full((M,8),.125,device='cuda',dtype=torch.float32)
 ids=torch.full((M,8),-1,device='cuda',dtype=torch.int32);x=torch.zeros((M,K),device='cuda',dtype=torch.bfloat16);routes=CompactRoutes(ids,E,cutoff);B0=torch.full((S,K),37.,device='cuda',dtype=torch.bfloat16);B1=torch.full((S,K),37.,device='cuda',dtype=torch.bfloat16);reference=torch.full((S,N),37.,device='cuda',dtype=torch.bfloat16);refact=torch.full_like(B1,37.);scatter_raw=torch.full_like(B0,37.);scatter_act=torch.full_like(B1,37.);scales=torch.linspace(.25,2.,E,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32)
 def run_a(post):
  light,heavy,_=split.partition_routes(ids,E,cutoff);alignments=[]
  for policy,part in [('light',light),('heavy',heavy)]:
   label='indexed_'+policy;align=moe_align_block_size(part,configs[label]['block_shape'][0],E,ignore_invalid_expert=True);alignments.append(align);ops.launch_kernel(configs=tables[label],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=reference,sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=8,valid_shape_m=S)
  if post:scale.scale_active_rows(reference,ids.flatten(),scales);silu_and_mul(reference,refact,expert_ids=ids.flatten(),expert_step=1)
  return alignments
 def run_b(post):
  routes.update(ids);routes.gather(x,B0)
  for policy,offset in [('light',routes.light),('heavy',routes.heavy)]:kernels['grouped_'+policy].run(candidate,B0,B1,offset,scales,locks)
 def capture(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):fn()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 down_a=reference.view(-1)[:S*K].view(S,K);down_b=B0.view(-1)[:S*K].view(S,K);finals={}
 def full_a():
  alignments=run_a(True)
  for policy,align in zip(('light','heavy'),alignments):
   label='w2_indexed_'+policy;assert configs[label]['block_shape'][0]==configs['indexed_'+policy]['block_shape'][0];ops.launch_kernel(configs=tables[label],inputs=refact,weight=down_layer.weight,weight_scale=down_layer.weight_scale,outputs=down_a,sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=1,valid_shape_m=S)
  finals['A']=finalizer.fused_post_expert_norm_weighted_sum(down_a.view(M,8,K),scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=down_scales)
 def full_b():
  run_b(True)
  for policy,offset in [('light',routes.light),('heavy',routes.heavy)]:
   label='w2_grouped_'+policy;ops.launch_kernel(configs=tables[label],inputs=B1,weight=down_layer.weight,weight_scale=down_layer.weight_scale,outputs=down_b,expert_layout=offset,locks=locks,top_k=1,valid_shape_m=S)
  routes.scatter(down_b,B1)
  finals['B']=finalizer.fused_post_expert_norm_weighted_sum(B1.view(M,8,K),scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=down_scales)
 if torch.cuda.mem_get_info()[0]<reserve:raise RuntimeError('Reserve exhausted after setup; abort before graph launches')
 stagegraphs={side:capture(fn) for side,fn in [('A',lambda:run_a(True)),('B',lambda:run_b(True))]}
 graphs={side:capture(fn) for side,fn in [('A',full_a),('B',full_b)]}
 assert B0.shape==B1.shape==(S,K) and B0.numel()+B1.numel()==2*S*K
 report['candidate_scratch_bytes']=(B0.numel()+B1.numel())*2;report['normal_scratch_bytes']=3*S*K*2
 assert finals['A'].data_ptr()!=finals['B'].data_ptr() and finals['A'].shape==finals['B'].shape==(M,K)
 def exact(a,b):return bool(torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)))
 for seed in range(3):
  for pattern in ('uniform','onehot','eighthot','last','remote','crossing'):
   rng=random.Random(seed+29);local=[]
   for t in range(M):
    row=rng.sample(range(256),8) if pattern=='uniform' else [0]+rng.sample(range(32,256),7) if pattern=='onehot' else rng.sample(range(8),8) if pattern=='eighthot' else [E-1]+rng.sample(range(32,256),7) if pattern=='last' else rng.sample(range(32,256),8) if pattern=='remote' else ([0]+rng.sample(range(32,256),7) if t<(15 if seed%2 else 17) else rng.sample(range(32,256),8))
    assert len(set(row))==8;local.extend(v if v<E else -1 for v in row)
   ids.copy_(torch.tensor(local,device='cuda',dtype=torch.int32).view(M,8));torch.manual_seed(seed+83);x.copy_(torch.randn_like(x)/20);active=ids.flatten()>=0;cpu=plan(local,E,cutoff)
   for buf in (B0,B1,reference,refact,scatter_raw,scatter_act):buf.fill_(37.)
   stagegraphs['A'].replay();stagegraphs['B'].replay();routes.scatter(B1,scatter_act);torch.cuda.synchronize()
   activation_checks={'activation_bitwise':exact(refact[active],scatter_act[active]),'activation_finite':bool(torch.isfinite(refact[active]).all() and torch.isfinite(scatter_act[active]).all()),'activation_inactive_sentinel':bool((refact[~active]==37).all() and (scatter_act[~active]==37).all()),'compact_activation_tail':bool((B1[cpu['total']:]==37).all())}
   assert all(activation_checks.values()),activation_checks
   finals['A'].fill_(float('nan'));finals['B'].fill_(float('nan'))
   scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True));graphs['A'].replay();graphs['B'].replay();torch.cuda.synchronize()
   checks={'down_bitwise':exact(down_a[active],B1[active]),'final_bitwise':exact(finals['A'],finals['B']),'down_finite':bool(torch.isfinite(down_a[active]).all() and torch.isfinite(B1[active]).all()),'final_finite':bool(torch.isfinite(finals['A']).all() and torch.isfinite(finals['B']).all()),'map_exact':routes.map.cpu().tolist()[:cpu['total']]==cpu['compact_to_slot'],'inverse_exact':routes.inverse.cpu().tolist()==cpu['slot_to_compact']}
   checks.update(activation_checks)
   if pattern=='remote':checks['empty_rank_final_zero']=bool((finals['A']==0).all() and (finals['B']==0).all())
   report['rows'].append({'seed':seed,'pattern':pattern,'active':cpu['total'],'checks':checks});save();assert all(checks.values()),report['rows'][-1]
 report['prior_endpoint_negative_receipt']='compact_e8_w13_01/results.json; no raw-width negative launch into half-width buffers'
 report['source_unchanged']=all(sha(Path(f))==h for f,h in report['source_hashes'].items());assert report['source_unchanged'];report['status']='compact_fused_two_buffer_gate_passed';report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save();print(json.dumps({'status':report['status'],'cases':len(report['rows'])}))
if __name__=='__main__':main()
