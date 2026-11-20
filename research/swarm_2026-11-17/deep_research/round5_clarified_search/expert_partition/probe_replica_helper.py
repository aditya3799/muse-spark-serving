"""Bounded small-geometry native Humming slot-copy correctness only."""
from pathlib import Path
import argparse,ast,copy,dataclasses,gc,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists());OLD=ROOT/'swarm_2026-11-16/moe';EXPORT=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not(a.out/'results.json').exists()
 r={'status':'STARTED','scope':'SM86 H=I256,M32,top8,physical32->33,policy32. Identical packed expert row copies and full helper; no timing, actual multi-rank collective or H200 claim.','memory':[],'checks':[],'cases':[]}
 def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 save()
 for d in (OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15'):sys.path.insert(0,str(d))
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 import torch
 from humming.layer import HummingMethod,HummingLayerMeta
 from fixture_tuning_cache import cached_threshold_for_preparation
 assert torch.cuda.get_device_capability()==(8,6)
 reserve=512*1024**2;budget=128*1024**2;entry=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 def memory(label,extra=0):
  free,total=torch.cuda.mem_get_info();peak=torch.cuda.max_memory_allocated()-entry;allocated=torch.cuda.memory_allocated()-entry
  row=dict(label=label,free=free,allocated_delta=allocated,peak_delta=peak,reserved=torch.cuda.memory_reserved(),estimated_next=extra);r['memory'].append(row);save()
  assert free>=reserve+extra and peak<=budget and allocated+extra<=budget,row
 memory('entry',24*1024**2)
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 scale=load('sglang.kernels.ops.moe.muse_spark_humming_scale',EXPORT/'kernels/ops/moe/muse_spark_humming_scale.py')
 split=load('sglang.kernels.ops.moe.muse_spark_humming_split',P/'humming_split_capacity_overlay.py')
 final=load('replica_scored_finalizer',EXPORT/'kernels/ops/moe/muse_spark_post_expert_finalize_triton.py')
 source=EXPORT/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py';tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('_prepare_humming_experts','_humming_expert_outputs_default','_humming_expert_outputs')];ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
 files=[Path(__file__),P/'humming_split_capacity_overlay.py',source,Path(scale.__file__),Path(final.__file__),ROOT/'optimization_marlin_tiles_2026-11-15/kernel_runtime.py',OLD/'phase2/fixture_tuning_cache.py',*sorted((OLD/'humming_vendor/humming').rglob('*.py')),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]
 r['source_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files};save()
 torch.manual_seed(98171);h=256;e=32;m=32;topk=8;raw=torch.nn.Module()
 for stage,n in [('w13',2*h),('w2',h)]:
  w=torch.randint(0,256,(e,n,h//2),device='cuda',dtype=torch.uint8)
  sf=torch.randint(40,81,(e,n,h//16),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
  gs=torch.linspace(.125,.5,e,device='cuda',dtype=torch.bfloat16)
  for suffix,t in [('weight',w),('weight_scale',sf),('weight_scale_2',gs)]:setattr(raw,stage+'_'+suffix,torch.nn.Parameter(t,requires_grad=False))
 memory('raw_fixture',16*1024**2)
 with cached_threshold_for_preparation(a.out):base=ns['_prepare_humming_experts'](raw,torch)
 del raw,w,sf,gs,t;gc.collect();torch.cuda.empty_cache();memory('native_packed_reference',16*1024**2)
 cand=torch.nn.Module();cand.humming_metas={}
 for stage in ('w13','w2'):
  meta=base.humming_metas[stage];cfg=dataclasses.replace(meta,num_experts=33)
  cand.humming_metas[stage]=HummingLayerMeta.from_layer_config(cfg,stage)
  for suffix in ('weight','weight_scale','post_scale'):
   value=getattr(base,stage+'_'+suffix);dest=torch.empty((33,)+value.shape[1:],device=value.device,dtype=value.dtype)
   dest[:32].copy_(value);dest[32].copy_(value[0])
   if suffix=='post_scale':cand.register_buffer(stage+'_'+suffix,dest)
   else:cand.register_parameter(stage+'_'+suffix,torch.nn.Parameter(dest,requires_grad=False))
 cand.register_buffer('locks',torch.zeros_like(base.locks));cand.fp32_configs=copy.deepcopy(base.fp32_configs);cand._research_policy_local_experts=32
 # Physical metadata differs, chosen compute configurations intentionally remain original.
 r['owned_configs']=base.fp32_configs;r['reference_plan']=split.choose_partition_configs(base,m*topk,256);r['candidate_plan']=split.choose_partition_configs(cand,m*topk,256,policy_experts=32)
 assert r['reference_plan']==r['candidate_plan'];r['adaptive_split_supported']=r['reference_plan'] is not None;save()
 pointers={stage+suffix:getattr(cand,stage+'_'+suffix).data_ptr()for stage in ('w13','w2')for suffix in ('weight','weight_scale','post_scale')}
 x=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);ids=torch.empty((m,topk),device='cuda',dtype=torch.int32);mapped=torch.empty_like(ids);left=torch.empty_like(ids);right=torch.empty_like(ids);scores=torch.empty((m,topk),device='cuda');norm=torch.linspace(.75,1.25,h,device='cuda',dtype=torch.bfloat16)
 def refresh(extra):
  for stage in ('w13','w2'):
   for suffix in ('weight','weight_scale','post_scale'):
    src=getattr(base,stage+'_'+suffix);dst=getattr(cand,stage+'_'+suffix);dst[:32].copy_(src);dst[32].copy_(src[extra])
 def inputs(pattern,generation):
  extra=31 if pattern=='last'else 0;refresh(extra)
  rows=[]
  for t in range(m):
   row=list(range(24,32))if pattern=='last'else list(range(8))
   if pattern=='remote' or(pattern=='invalid_tail'and t>=27):row=[-1]*8
   rows.append(row)
  cpu=torch.tensor(rows,dtype=torch.int32);cm=cpu.clone()
  target=(cpu==extra)
  if pattern=='split':target=target & (torch.arange(m)[:,None]%2==1)
  cm[target]=32
  cl=cm.clone();cr=torch.full_like(cm,-1);isrep=cm==32;cl[isrep]=-1;cr[isrep]=32
  assert torch.equal(((cl>=0).to(torch.int32)+(cr>=0).to(torch.int32)),(cpu>=0).to(torch.int32))
  ids.copy_(cpu);mapped.copy_(cm);left.copy_(cl);right.copy_(cr)
  torch.manual_seed(181+generation);x.copy_(torch.randn_like(x)*.125);scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
  return extra
 def run(layer,route):
  down,_=ns['_humming_expert_outputs'](layer,x,route,torch,defer_downscale=True,global_experts=256)
  out=final.fused_post_expert_norm_weighted_sum(down,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=route,expert_output_scale=layer.w2_post_scale)
  return down,out
 def replicated():
  da,oa=run(cand,left);db,ob=run(cand,right)
  assert da.data_ptr()!=db.data_ptr() and oa.data_ptr()!=ob.data_ptr()
  merged=torch.where((left>=0).unsqueeze(-1),da,db)
  merged_out=final.fused_post_expert_norm_weighted_sum(merged,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=base.w2_post_scale)
  ep_sum=(oa.float()+ob.float()).to(torch.bfloat16) # Local emulation only.
  return merged,merged_out,ep_sum
 def capture(fn):
  memory('before_capture',16*1024**2)
  for _ in range(2):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn()
  memory('after_capture');return g,out
 inputs('migrate',0)
 graphs={};outputs={}
 for key,fn in [('reference',lambda:run(base,ids)),('mapped',lambda:run(cand,mapped)),('two_rank_emulated',replicated)]:graphs[key],outputs[key]=capture(fn)
 assert len({out[1].data_ptr()for out in outputs.values()})==3
 def compare(a,b,label,strict=False):
  a=a.contiguous();b=b.contiguous();finite=bool(torch.isfinite(a).all()and torch.isfinite(b).all());eq=torch.equal(a.view(torch.int16),b.view(torch.int16));err=float((a.float()-b.float()).abs().max())if a.numel()else 0
  row=dict(label=label,finite=finite,bitwise=eq,elements=a.numel(),max_abs=err,strict_copy_gate=strict);r['checks'].append(row);save();assert finite,label
  if strict:assert eq,label
 for gen in range(2):
  if gen:
   for stage in ('w13','w2'):
    getattr(base,stage+'_weight').bitwise_xor_(0x08800880)
    sf=getattr(base,stage+'_weight_scale');sf.view(torch.uint8).copy_(((sf.view(torch.uint8).to(torch.int16)+7-40)%41+40).to(torch.uint8));getattr(base,stage+'_post_scale').mul_(.75)
  for pattern in ['migrate','split','last','invalid_tail','remote']:
   extra=inputs(pattern,gen)
   assert pointers=={stage+suffix:getattr(cand,stage+'_'+suffix).data_ptr()for stage in ('w13','w2')for suffix in ('weight','weight_scale','post_scale')}
   for value in outputs.values():
    for tensor in value:tensor.fill_(float('nan'))
   for g in graphs.values():g.replay()
   torch.cuda.synchronize();active=ids>=0
   if bool(active.any()):
    compare(outputs['mapped'][0][active],outputs['reference'][0][active],[gen,pattern,'mapped_raw'],strict=pattern!='split')
    compare(outputs['two_rank_emulated'][0][active],outputs['reference'][0][active],[gen,pattern,'merged_raw'],strict=pattern!='split')
   compare(outputs['mapped'][1],outputs['reference'][1],[gen,pattern,'mapped_final'],strict=pattern!='split')
   compare(outputs['two_rank_emulated'][1],outputs['reference'][1],[gen,pattern,'merged_final'],strict=pattern!='split')
   compare(outputs['two_rank_emulated'][2],outputs['reference'][1],[gen,pattern,'emulated_EP_sum'],strict=False)
   inactive=~active.any(-1);assert all(bool((v[1][inactive]==0).all())for v in outputs.values())
   r['cases'].append(dict(generation=gen,pattern=pattern,replica_logical_expert=extra));memory('case_complete')
 r['status']='PASS_NATIVE_PACKED_SLOT_COPY_AND_HELPER_WITH_REORDER_DIAGNOSTICS';r['limits']=['No actual rank collective or peer-copy protocol; local emulation only.','Split-owner numerical differences are recorded, not called rule failures; whole-expert row copy is strict.','Fixed original compute tables; physical33 metadata regenerated, default table regeneration not performance qualified.','No timing/H200/full-model or activated serving path.'];save();print(json.dumps({'status':r['status'],'cases':len(r['cases']),'checks':len(r['checks'])}))
if __name__=='__main__':main()
