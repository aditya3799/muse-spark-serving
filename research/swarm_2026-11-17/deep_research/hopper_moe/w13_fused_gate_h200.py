"""Prepared H200-only fused W13 numerical chain gate; no timing or submissions."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os,sys
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'swarm_2026-11-16/moe';COORD=P.parent/'coordination';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
from w13_h200_contract import approved_job,require_h200
from virtual_k_gate_contract import route_arrays

def main():
 pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--seeds',type=int,default=3);pa.add_argument('--jobs',nargs='+',default=['w13_m8_fused','w13_m256_fused']);a=pa.parse_args()
 if a.out.exists():raise ValueError('Fresh output directory required')
 if not 1<=a.seeds<=10:raise ValueError('Use 1..10 seeds')
 jobs=[approved_job(j) for j in a.jobs]
 if len(set(a.jobs))!=len(jobs):raise ValueError('Duplicate job')
 for path in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(path))
 import torch
 prop=torch.cuda.get_device_properties();require_h200(prop.major,prop.minor,prop.name)
 if os.environ.get('MAX_JOBS')!='1' or os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD')!='1':raise RuntimeError('Serial compiler environment required')
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 from humming.layer import HummingMethod
 from fixture_tuning_cache import cached_threshold_for_preparation
 from sglang.kernels.ops.activation.activation import silu_and_mul
 from w13_h200_precompiled import prepare_h200
 scale_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_humming_scale.py'
 final_path=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
 def load_module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 scale_active_rows=load_module('h200_gate_scale',scale_path).scale_active_rows
 finalize=load_module('h200_gate_finalize',final_path).fused_post_expert_norm_weighted_sum
 src=EXPORT/'python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py';stock=ROOT/'sglang/python/sglang/test/test_marlin_utils.py'
 def extract(path,name,ns):
  fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns);return ns[name]
 prepare=extract(src,'_prepare_humming_experts',{});generate=extract(stock,'make_nvfp4_weight_and_ref',{'torch':torch})
 a.out.mkdir(parents=True)
 files=[Path(__file__),P/'w13_h200_precompiled.py',P/'w13_h200_contract.py',P/'virtual_k_gate_contract.py',P/'w13_fused_jobs.json',P/'split_issue_jobs.json',P/'offline_w13_fused/results.json',COORD/'w13_fused_epilogue.py',scale_path,final_path,src,stock,*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 report={'status':'running','scope':'Two fixed cp.async W13 SM90 geometries, canonical/interleaved preparation, raw/scaled/SiLU comparison, W2/finalizer and changed-input graph checks. No serving or timing claim.','gpu':prop.name,'SM':[prop.major,prop.minor],'source_hashes':{str(f):sha(f) for f in files},'compiled_fused_receipt':json.loads((P/'offline_w13_fused/results.json').read_text())['cases'],'checks':[],'nonfinite_policy':'Fixtures are finite. Any nonfinite compared output fails; NaN/Inf payload equivalence is not asserted.'}
 def save():
  tmp=a.out/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');tmp.replace(a.out/'results.json')
 def check(x,y,**meta):
  finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all());same=bool(torch.equal(x.contiguous().view(torch.int16),y.contiguous().view(torch.int16)))
  item=dict(meta,finite=finite,bitwise=same);report['checks'].append(item);save()
  if not (finite and same):raise AssertionError(item)
 def capture(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  return g,out
 save();torch.manual_seed(771);h=4096;e=32;raw=torch.nn.Module()
 for stage,n in (('w13',h*2),('w2',h)):
  weights=[];scales=[];globals_=[]
  for expert in range(e):
   w,s,g,ref=generate(n,h,torch.bfloat16);weights.append(w);scales.append(s);globals_.append(g);del ref
  for suffix,values in (('weight',weights),('weight_scale',scales),('weight_scale_2',globals_)):setattr(raw,stage+'_'+suffix,torch.nn.Parameter(torch.stack(values),requires_grad=False))
  del weights,scales,globals_,w,s,g,values
  getattr(raw,stage+'_weight_scale_2').mul_(torch.linspace(.25,2.,e,device='cuda'))
 perm=torch.stack((torch.arange(h,device='cuda'),torch.arange(h,2*h,device='cuda')),dim=1).flatten();inverse=torch.argsort(perm)
 with cached_threshold_for_preparation(a.out):
  baseline=prepare(raw,torch)
  raw.w13_weight=torch.nn.Parameter(raw.w13_weight[:,perm].contiguous(),requires_grad=False)
  raw.w13_weight_scale=torch.nn.Parameter(raw.w13_weight_scale.view(torch.uint8)[:,perm].contiguous().view(torch.float8_e4m3fn),requires_grad=False)
  candidate=prepare(raw,torch)
 del raw
 if not torch.equal(baseline.w2_weight,candidate.w2_weight) or not torch.equal(baseline.w2_weight_scale.view(torch.uint8),candidate.w2_weight_scale.view(torch.uint8)):raise AssertionError('W2 preparation changed')
 norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16)
 initial_scales={s:getattr(baseline,s+'_post_scale').clone() for s in ('w13','w2')}
 for job in jobs:
  c=job['configuration'];m=job['tokens'];slots=m*8
  reference_jobs=json.loads((P/'split_issue_jobs.json').read_text())['jobs']
  cfg2=next(j['configuration'] for j in reference_jobs if j['stage']=='w2' and j['policy']=='control' and j['tokens']==m)
  if not cfg2.get('use_cp_async') or any(v for k,v in cfg2.items() if k.startswith('use_tma')):raise ValueError('W2 control must use the compiled cp.async policy')
  # Only W13 uses the custom driver ABI. W2 uses an existing compiled cp.async control and the canonical launcher.
  fused=prepare_h200(job['job_id']);(a.out/(job['job_id']+'.cu')).write_text(fused.code,encoding='utf-8')
  ids=torch.empty((m,8),device='cuda',dtype=torch.int32);x=torch.zeros((m,h),device='cuda',dtype=torch.bfloat16)
  def allocate_alignment(block):
   r=route_arrays(m,e,block,'uniform',0)
   return [torch.tensor(r['sorted_ids'],device='cuda',dtype=torch.int32),torch.tensor(r['expert_ids'],device='cuda',dtype=torch.int32),torch.tensor([r['padded']],device='cuda',dtype=torch.int32)]
  aligned=allocate_alignment(c['block_shape'][0]);aligned2=allocate_alignment(cfg2['block_shape'][0])
  def change_routes(pattern,seed):
   for block,arr in ((c['block_shape'][0],aligned),(cfg2['block_shape'][0],aligned2)):
    r=route_arrays(m,e,block,pattern,seed)
    arr[0].copy_(torch.tensor(r['sorted_ids'],device='cuda',dtype=torch.int32));arr[1].copy_(torch.tensor(r['expert_ids'],device='cuda',dtype=torch.int32));arr[2].fill_(r['padded'])
   ids.copy_(torch.tensor(r['local_ids'],device='cuda',dtype=torch.int32).view(m,8))
  change_routes('uniform',0)
  outputs=[torch.full((slots,h*2),37.,device='cuda',dtype=torch.bfloat16) for _ in range(2)];activation=torch.full((slots,h),37.,device='cuda',dtype=torch.bfloat16);fused_output=torch.full_like(activation,37.)
  downs=[torch.full_like(activation,37.),torch.full_like(activation,37.)];scores=torch.full((m,8),.125,device='cuda')
  def raw_call(layer,target):
   return HummingMethod.forward_layer(layer,inputs=x,outputs=target,top_k=8,sublayer_name='w13',tuning_config=c,sorted_ids=aligned[0],expert_ids=aligned[1],num_tokens_padded=aligned[2],valid_shape_m=slots,compute_config={'gemm_type':'indexed','use_f16_accum':False})
  def component_a():
   raw_call(baseline,outputs[0]);scale_active_rows(outputs[0],ids.flatten(),baseline.w13_post_scale);silu_and_mul(outputs[0],activation,expert_ids=ids.flatten(),expert_step=1);return activation
  def component_b():return fused.run(candidate,x,fused_output,ids,aligned)
  def down(layer,value,target):
   HummingMethod.forward_layer(layer,inputs=value,outputs=target,top_k=1,sublayer_name='w2',tuning_config=cfg2,sorted_ids=aligned2[0],expert_ids=aligned2[1],num_tokens_padded=aligned2[2],valid_shape_m=slots,compute_config={'gemm_type':'indexed','use_f16_accum':False})
   scale_active_rows(target,ids.flatten(),layer.w2_post_scale)
   return finalize(target.view(m,8,h),scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids)
  graph_a,final_graph_a=capture(lambda:down(baseline,component_a(),downs[0]));graph_b,final_graph_b=capture(lambda:down(candidate,component_b(),downs[1]))
  for seed in range(a.seeds):
   for pattern in ('uniform','hot_expert','mixed_duplicate','all_remote'):
    change_routes(pattern,seed);active=ids.flatten()>=0
    for stage in ('w13','w2'):
     values=(initial_scales[stage].float()*(1.+seed*.125)).to(torch.bfloat16)
     for layer in (baseline,candidate):getattr(layer,stage+'_post_scale').copy_(values)
    torch.manual_seed(771+m+seed);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
    for profile in ('random','subnormal','signed_zero'):
     if profile=='random':x.copy_(torch.randn_like(x)/20)
     elif profile=='subnormal':x.copy_(torch.arange(h,device='cuda').remainder(2).mul(2).sub(1).float().mul(2.**-130).to(torch.bfloat16).expand_as(x))
     else:x.zero_();x[:,::2]=-0.
     meta={'job':job['job_id'],'seed':seed,'pattern':pattern,'profile':profile}
     raw_call(baseline,outputs[0]);raw_call(candidate,outputs[1]);aa=outputs[0][active];bb=outputs[1][active][:,inverse];check(aa,bb,stage='raw_gemm',**meta)
     scales=baseline.w13_post_scale[ids.flatten()[active].long(),None];aa=(aa.float()*scales.float()).to(torch.bfloat16);bb=(bb.float()*scales.float()).to(torch.bfloat16);check(aa,bb,stage='scaled_bf16',**meta)
     oa=torch.empty((aa.shape[0],h),device='cuda',dtype=torch.bfloat16);ob=torch.empty_like(oa)
     if aa.shape[0]:silu_and_mul(aa,oa);silu_and_mul(bb,ob)
     check(oa,ob,stage='stock_silu',**meta)
     activation.fill_(37.);fused_output.fill_(37.);component_a();component_b();check(oa,fused_output[active],stage='fused_epilogue',**meta);check(activation[active],fused_output[active],stage='component',**meta)
     if not bool((fused_output[~active]==37.).all()):raise AssertionError('Inactive fused row overwritten')
     final_a=down(baseline,activation,downs[0]).clone();final_b=down(candidate,fused_output,downs[1]).clone();check(downs[0][active],downs[1][active],stage='scaled_w2',**meta);check(final_a,final_b,stage='finalizer',**meta)
     graph_a.replay();graph_b.replay();torch.cuda.synchronize();check(final_graph_a,final_a,stage='changed_input_graph_a',**meta);check(final_graph_b,final_b,stage='changed_input_graph_b',**meta)
     if not bool((fused_output[~active]==37.).all()):raise AssertionError('Captured fused kernel overwrote inactive row')
  del graph_a,graph_b,outputs,activation,fused_output,downs,x,ids,aligned,aligned2,fused
  torch.cuda.synchronize()
 report['source_unchanged']=all(sha(Path(f))==digest for f,digest in report['source_hashes'].items());assert report['source_unchanged'];report['status']='H200_FIXED_CONFIG_FINITE_CHAIN_GATE_PASSED';save();print(json.dumps({'status':report['status'],'checks':len(report['checks'])}),flush=True)
if __name__=='__main__':main()
