"""BM8/24 shape and changed-weight gate. Fixed small geometry, no timing."""
from pathlib import Path
import argparse,hashlib,json,sys,random
P=Path(__file__).resolve().parent;ROOT=P.parents[3];OLD=ROOT/'swarm_2026-11-16/moe'
for path in [OLD/'humming_vendor',OLD/'phase2',ROOT/'optimization_marlin_tiles_2026-11-15']:sys.path.insert(0,str(path))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);assert not(a.out/'results.json').exists()
 import torch
 from kernel_runtime import bootstrap
 bootstrap(ROOT/'sglang')
 from humming import ops,dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingWeightSchema,HummingInputSchema
 from humming.kernel.humming import HummingKernel
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 from swap_ab_humming_runtime import SwapHummingKernel,Compiler
 assert torch.cuda.get_device_capability()==(8,6)
 free,_=torch.cuda.mem_get_info();assert free>1024*1024**2,'512MiB allocation estimate+512MiB reserve'
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 files=[Path(__file__),P/'swap_ab_humming.cuh',P/'swap_ab_humming_codegen.py',P/'swap_ab_humming_runtime.py',P/'SM86_COMPACT_E8_W13_JOBS.json',*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh')),*sorted((OLD/'humming_vendor/humming').rglob('*.py'))]
 r=dict(status='RUNNING',scope='SM86 E1/E8 N256 K4096 M64topK8 geometry diagnostic: BM8/24 directcanonical swap, BK128/WK64 twoKbanks; baselineBM16 separatelyaligned. Changedcaptured weight/scalepointers contents, no timing/fullhelper/nativeWGMMA claim.',sources={str(f):sha(f)for f in files},kernels=[],rows=[],weight_generations=[],memory=dict(free=free,reserve=512*1024**2,estimated=512*1024**2))
 def save():(a.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 def cmp(x,y):
  xx=x.contiguous();yy=y.contiguous();neq=xx.view(torch.int16)!=yy.view(torch.int16)
  return dict(bitwise=not bool(neq.any()),different=int(neq.sum()),finite=bool(torch.isfinite(xx).all()and torch.isfinite(yy).all()),elements=xx.numel(),max_abs=float((xx.float()-yy.float()).abs().max())if xx.numel()else 0.)
 class Base(HummingKernel):
  def __new__(cls,*args,**kw):o=object.__new__(cls);o.inited=False;o.cubin_loaded=False;return o
  @staticmethod
  def _get_compiler():return Compiler
 n=256;k=4096;m=64;capacity=m*8
 base=json.loads((P/'SM86_COMPACT_E8_W13_JOBS.json').read_text())['jobs'][0]['configuration']
 schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
 def weights(e,gen):
  torch.manual_seed(829+gen);raw=torch.randint(0,256,(e,n,k//2),device='cuda',dtype=torch.uint8)
  finite=torch.tensor([v for v in range(256)if(v&127)!=127],device='cuda',dtype=torch.uint8)
  idx=(torch.arange(e*n*k//16,device='cuda').reshape(e,n,k//16)*19+gen*31)%254;sf=finite[idx].view(torch.float8_e4m3fn).contiguous()
  layer=torch.nn.Module();layer.register_parameter('weight',torch.nn.Parameter(raw.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf,requires_grad=False))
  HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True)
  word=layer.weight.to(torch.int64)&0xffffffff;canonical=((word&0xf00ff00f)|((word&0x00f000f0)<<4)|((word&0x0f000f00)>>4)).to(torch.int32).contiguous()
  return layer,canonical
 for e in [1,8]:
  configs={};kernels={};tables={};variants={'pinned':(16,False),'bm8_static':(8,False),'bm8_semi':(8,True),'bm24_static':(24,False),'bm24_semi':(24,True)}
  for name,(bm,semi)in variants.items():
   cfg=dict(base,shape_n=n,shape_k=k,num_experts=e,block_shape=[bm,256,128],warp_shape=[bm,64,64]);configs[name]=cfg
   ker=Base(**cfg)if name=='pinned'else SwapHummingKernel(**cfg,research_direct=True,research_semistatic=semi);kernels[name]=ker;tables[name]=torch.tensor([0,1<<30,ker.kernel_id,0],dtype=torch.int64)
   dest=a.out/('e%d_'%e+name);dest.mkdir();(dest/'kernel.cu').write_text(ker.code);cb=Path(ker.kernel_filename);(dest/'kernel.sm86.cubin').write_bytes(cb.read_bytes());res=[]
   for fn in ['stdout.log','stderr.log','cmdline.json']:
    if(cb.parent/fn).exists():(dest/fn).write_bytes((cb.parent/fn).read_bytes());res.extend(v for v in(cb.parent/fn).read_text(errors='replace').splitlines()if any(t in v for t in ['Used ','spill','stack frame']))
   r['kernels'].append(dict(e=e,name=name,configuration=cfg,cubin_sha256=sha(cb),resources=res,shared_bytes=ker.estimated_smem_size));save()
  layer,canonical=weights(e,0);initial_ptrs=(layer.weight.data_ptr(),layer.weight_scale.data_ptr(),canonical.data_ptr());ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32);x=torch.zeros(m,k,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32);out={name:torch.full((capacity,n),37.,device='cuda',dtype=torch.bfloat16)for name in variants};assert len(set(t.data_ptr()for t in out.values()))==len(out)
  assert canonical.shape==layer.weight.shape==(e,k//16,n*2)and layer.weight_scale.shape==(e,k//16,n)
  def run(name):
   align=moe_align_block_size(ids,configs[name]['block_shape'][0],e,ignore_invalid_expert=True)
   return ops.launch_kernel(configs=tables[name],inputs=x,weight=layer.weight if name=='pinned'else canonical,weight_scale=layer.weight_scale,outputs=out[name],sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=8,valid_shape_m=capacity)
  graphs={}
  for name in variants:
   for _ in range(2):run(name)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):run(name)
   graphs[name]=g
  assert torch.cuda.mem_get_info()[0]>512*1024**2
  for generation in range(3):
   changed=True
   if generation:
    newlayer,newcanonical=weights(e,generation);changed=bool((newcanonical!=canonical).any())and bool((newlayer.weight_scale.view(torch.uint8)!=layer.weight_scale.view(torch.uint8)).any());assert changed
    layer.weight.copy_(newlayer.weight);layer.weight_scale.copy_(newlayer.weight_scale);canonical.copy_(newcanonical);del newlayer,newcanonical
   assert initial_ptrs==(layer.weight.data_ptr(),layer.weight_scale.data_ptr(),canonical.data_ptr());r['weight_generations'].append(dict(e=e,generation=generation,pointers_stable=True,weight_and_scale_changed=changed));save()
   patterns=list(range(0,1))+[1,7,8,9,15,16,17,23,24,25,31,32,33,47,48,49,64]+['last','uniform','mixed']
   for pat in patterns:
    rng=random.Random(41+generation);local=[];globalrows=[]
    for token in range(m):
     chosen=[]
     if isinstance(pat,int)and token<pat:chosen=[0]
     elif pat=='last':chosen=[e-1]
     elif pat=='uniform':chosen=rng.sample(range(64),8)
     elif pat=='mixed':chosen=[0]if token<49 else [];chosen+=([e-1]if e>1 and token<25 else [])
     if len(chosen)<8:chosen+=rng.sample(range(8,64),8-len(chosen))
     rng.shuffle(chosen);assert len(chosen)==len(set(chosen))==8
     globalrows.append(chosen);local.extend(v if v<e else -1 for v in chosen)
    ids.copy_(torch.tensor(local,device='cuda',dtype=torch.int32).view(m,8));torch.manual_seed(713+generation)
    if generation==0:x.copy_((torch.arange(m*k,device='cuda').reshape(m,k)%2*2-1).float()/128)
    elif generation==1:x.copy_(torch.randn_like(x)/128)
    else:x.copy_(((torch.arange(k,device='cuda')%16-8)[None,:].expand(m,-1)/1024).to(torch.bfloat16))
    # Outside captured graph, validate each real alignment tile is expert-homogeneous and valid-prefix, including trailing padding.
    for bm in [8,16,24]:
     al=moe_align_block_size(ids,bm,e,ignore_invalid_expert=True);pad=int(al[2].item());sortedids=al[0][:pad].cpu().tolist();expertids=al[1][:pad//bm].cpu().tolist();seen=[]
     for tile in range(pad//bm):
      invalid=False
      for slot in sortedids[tile*bm:(tile+1)*bm]:
       if slot>=capacity:invalid=True
       else:assert not invalid and local[slot]==expertids[tile];seen.append(slot)
     assert sorted(seen)==[i for i,v in enumerate(local)if v>=0]
    for t in out.values():t.fill_(37.)
    for g in graphs.values():g.replay()
    torch.cuda.synchronize();active=ids.flatten()>=0;post=torch.tensor([.125,.75,1.375][generation],device='cuda',dtype=torch.bfloat16);checks={}
    for name in variants:
     if name=='pinned':continue
     checks[name]=dict(raw=cmp(out[name][active],out['pinned'][active]),post=cmp(out[name][active]*post,out['pinned'][active]*post),inactive=bool((out[name][~active]==37).all()))
    row=dict(e=e,generation=generation,pattern=pat,counts=[local.count(i)for i in range(e)],checks=checks,baseline_inactive=bool((out['pinned'][~active]==37).all()));r['rows'].append(row);save()
    assert row['baseline_inactive']and all(v['inactive']and v['raw']['finite']and v['post']['finite']and v['raw']['bitwise']and v['post']['bitwise']for v in checks.values()),row
  del graphs,kernels,tables,layer,canonical,ids,x,locks,out
 r.update(status='COMPLETE_TILE_AND_CHANGED_WEIGHT_GATE',sources_unchanged=all(sha(f)==h for f,h in r['sources'].items()),peak_allocated=torch.cuda.max_memory_allocated());save();assert r['sources_unchanged'];print(json.dumps(dict(status=r['status'],cases=len(r['rows']),kernels=len(r['kernels']))),flush=True)
if __name__=='__main__':main()
