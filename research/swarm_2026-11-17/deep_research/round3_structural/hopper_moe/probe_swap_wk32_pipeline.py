"""Production-style cp.async E1 mapping gate. No performance/helper claim."""
from pathlib import Path
import argparse,hashlib,json,sys,random
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe'
for path in [OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15']:sys.path.insert(0,str(path))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);assert not(args.out/'results.json').exists()
 import torch
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 from humming import ops,dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingWeightSchema,HummingInputSchema
 from humming.kernel.humming import HummingKernel
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 from swap_ab_humming_runtime_v3 import SwapHummingKernel,Compiler
 assert torch.cuda.get_device_capability()==(8,6)
 free,_=torch.cuda.mem_get_info();assert free>768*1024**2
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 files=[Path(__file__),P/'wk32_engineered.py',P/'swap_ab_humming_v3.cuh',P/'swap_ab_humming_codegen_v3.py',P/'swap_ab_humming_runtime_v3.py',P/'SM86_COMPACT_E8_W13_JOBS.json',*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh')),*sorted((OLD/'humming_vendor/humming').rglob('*.py'))]
 r=dict(status='RUNNING',scope='E1 N256 K4096 BM16 BK64 WK32 original cp.async/scheduler/Kbank reduction, canonical packedWGMMA NVFP4 one candidate copy; same-precision arithmetic gate, no timing/serving/Hopper qualification.',sources={str(f):sha(f)for f in files},kernels=[],rows=[],preflight=dict(free=free,reserve=512*1024**2,estimated=256*1024**2))
 def save():(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 def compare(a,b):
  av=a.contiguous();bv=b.contiguous();neq=av.view(torch.int16)!=bv.view(torch.int16)
  return dict(bitwise=not bool(neq.any()),different=int(neq.sum()),elements=av.numel(),finite=bool(torch.isfinite(av).all()and torch.isfinite(bv).all()),max_abs=float((av.float()-bv.float()).abs().max())if av.numel()else 0.)
 class Baseline(HummingKernel):
  def __new__(cls,*a,**kw):obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
  @staticmethod
  def _get_compiler():return Compiler
 base=json.loads((P/'SM86_COMPACT_E8_W13_JOBS.json').read_text())['jobs'][0]['configuration'];n=256;k=4096;m=24;capacity=m*8
 cfg=dict(base,shape_n=n,shape_k=k,num_experts=1,block_shape=[16,256,64],warp_shape=[16,64,32])
 kernels={};tables={};variants={'pinned':None,'remap_static':(False,False),'direct_static':(True,False),'direct_semistatic':(True,True)}
 for name,opts in variants.items():
  ker=Baseline(**cfg)if opts is None else SwapHummingKernel(**cfg,research_direct=opts[0],research_semistatic=opts[1]);kernels[name]=ker;tables[name]=torch.tensor([0,1<<30,ker.kernel_id,0],dtype=torch.int64)
  d=args.out/name;d.mkdir();(d/'kernel.cu').write_text(ker.code);cb=Path(ker.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cb.read_bytes());resources=[]
  for fn in ['stdout.log','stderr.log','cmdline.json']:
   if(cb.parent/fn).exists():(d/fn).write_bytes((cb.parent/fn).read_bytes());resources.extend(v for v in(cb.parent/fn).read_text(errors='replace').splitlines()if any(t in v for t in ['Used ','spill','stack frame']))
  r['kernels'].append(dict(name=name,configuration=cfg,research=getattr(ker,'research',None),cubin_sha256=sha(cb),resources=resources));save()
 codes=(torch.arange(n*k,device='cuda').reshape(n,k)*7+torch.arange(n,device='cuda')[:,None])%16
 packed=(codes[:,::2]|(codes[:,1::2]<<4)).to(torch.uint8).unsqueeze(0).contiguous();finitecodes=torch.tensor([v for v in range(256)if(v&127)!=127],device='cuda',dtype=torch.uint8)
 idx=(torch.arange(n*(k//16),device='cuda').reshape(n,k//16)*19+torch.arange(n,device='cuda')[:,None]*31)%254
 sf=finitecodes[idx].view(torch.float8_e4m3fn).unsqueeze(0).contiguous();assert torch.unique(sf.view(torch.uint8)).numel()==254
 layer=torch.nn.Module();layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf.clone(),requires_grad=False))
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
 HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=1,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True)
 word=layer.weight.to(torch.int64)&0xffffffff;canonical=((word&0xf00ff00f)|((word&0x00f000f0)<<4)|((word&0x0f000f00)>>4)).to(torch.int32).contiguous();del word
 assert canonical.shape==layer.weight.shape==(1,k//16,n*2)and canonical.data_ptr()!=layer.weight.data_ptr();r['candidate_weight_layout']='pinned_WGMMA_transposed_B4_A16_group16';r['reference_copy_validation_only']=True
 ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32);x=torch.zeros(m,k,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32);out={name:torch.full((capacity,n),37.,device='cuda',dtype=torch.bfloat16)for name in variants};assert len(set(t.data_ptr()for t in out.values()))==len(out)
 def run(name):
  align=moe_align_block_size(ids,16,1,ignore_invalid_expert=True)
  return ops.launch_kernel(configs=tables[name],inputs=x,weight=layer.weight if name=='pinned'else canonical,weight_scale=layer.weight_scale,outputs=out[name],sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=8,valid_shape_m=capacity)
 graphs={}
 for name in variants:
  for _ in range(2):run(name)
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):run(name)
  graphs[name]=g
 assert torch.cuda.mem_get_info()[0]>512*1024**2
 for generation,count in enumerate([0,1,7,8,9,15,16,17,23,24,9,0,1]):
  rng=random.Random(829+generation);tokens=list(range(m));rng.shuffle(tokens);valid=[t*8+(t+generation)%8 for t in sorted(tokens[:count])];assert len(valid)==len(set(valid))==count
  host=[-1]*capacity
  for slot in valid:host[slot]=0
  ids.copy_(torch.tensor(host,device='cuda',dtype=torch.int32).reshape(m,8));torch.manual_seed(481+generation)
  if generation==12:
   from wk32_engineered import set_engineered
   r['engineered_contract']=set_engineered(layer,canonical,x,n,k)
  elif generation%3==0:x.copy_((torch.arange(m*k,device='cuda').reshape(m,k)%2*2-1).float()/128)
  elif generation%3==1:x.copy_(torch.randn_like(x)/128)
  else:x.copy_(((torch.arange(k,device='cuda')%16-8)[None,:].expand(m,-1)/1024).to(torch.bfloat16))
  # Alignment contract sampled outside captured kernel. No host route count is used by captured launches.
  alignment=moe_align_block_size(ids,16,1,ignore_invalid_expert=True);padded=int(alignment[2].item());ordered=alignment[0][:padded].cpu().tolist();assert sorted(v for v in ordered if v<capacity)==valid
  for start in range(0,padded,16):
   row=ordered[start:start+16];seen_invalid=False
   for v in row:
    if v>=capacity:seen_invalid=True
    else:assert not seen_invalid,'holey aligned block is forbidden'
  for t in out.values():t.fill_(37.)
  for g in graphs.values():g.replay()
  torch.cuda.synchronize();active=ids.flatten()==0;post=torch.tensor([.125,.75,1.375,2.][generation%4],device='cuda',dtype=torch.bfloat16)
  checks={}
  for name in variants:
   if name=='pinned':continue
   checks[name]=dict(raw=compare(out[name][active],out['pinned'][active]),post=compare(out[name][active]*post,out['pinned'][active]*post),inactive=bool((out[name][~active]==37).all()))
  row=dict(generation=generation,count=count,padded=padded,checks=checks,baseline_inactive=bool((out['pinned'][~active]==37).all()));r['rows'].append(row);save()
  if generation==12:
   assert all(bool((v[active]==8).all())for v in out.values());r['engineered_wrong_topology_rejected']=True;save()
  assert row['baseline_inactive']and all(v['inactive']and v['raw']['finite']and v['post']['finite']for v in checks.values()),row
  assert all(v['raw']['bitwise']and v['post']['bitwise']for v in checks.values()),row
 r.update(status='COMPLETE_PIPELINE_MAPPING_GATE',sources_unchanged=all(sha(f)==h for f,h in r['sources'].items()),peak_allocated=torch.cuda.max_memory_allocated());save();assert r['sources_unchanged'];print(json.dumps(dict(status=r['status'],cases=len(r['rows']),peak=r['peak_allocated'])),flush=True)
if __name__=='__main__':main()
