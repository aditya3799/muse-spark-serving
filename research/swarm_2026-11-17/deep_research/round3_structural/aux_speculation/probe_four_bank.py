"""E1 small-route arithmetic falsification; no performance or serving claim."""
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
 from humming.jit.compiler import NVRTCCompiler
 from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
 from four_bank_probe import Probe
 assert torch.cuda.get_device_capability()==(8,6)
 free,total=torch.cuda.mem_get_info();assert free>768*1024**2,'512MiB reserve +256MiB estimate'
 sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 source_files=[Path(__file__),P/'four_bank_probe.py',P/'four_bank_probe.cuh',P/'test_four_bank_reducer.py',P/'FOUR_BANK_REDUCER_CONTRACT.json',P/'FOUR_BANK_REFERENCE_CONFIG.json',*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh')),*sorted((OLD/'humming_vendor/humming').rglob('*.py'))]
 r=dict(status='RUNNING',scope='Four-bank E1 N256 K256/4096 direct-global canonical packed arithmetic microgate; pinned ordinary MMA comparator BN128/BK256/WK64 (two output tiles), not native SM90 WGMMA. Engineered bank sums raw2/x4=8 and raw1/x4=4; negative adjacent/left-fold kernels. No pipeline, performance, or H200 claim.',sources={str(f):sha(f)for f in source_files},rows=[],kernels=[],operand_checks=[],routes=[],memory=dict(free=free,reserve=512*1024**2),all_finite_scale_codes=254,signed_fp4_codes=16)
 def save():(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 def recordkernel(obj,label,**meta):
  d=args.out/label;d.mkdir();(d/'kernel.cu').write_text(obj.code);cb=Path(obj.kernel_filename);(d/'kernel.sm86.cubin').write_bytes(cb.read_bytes());resources=[]
  for fn in ['stdout.log','stderr.log','cmdline.json']:
   if(cb.parent/fn).exists():(d/fn).write_bytes((cb.parent/fn).read_bytes());resources.extend(v for v in (cb.parent/fn).read_text(errors='replace').splitlines()if any(s in v for s in ['Used ','spill','stack frame']))
  r['kernels'].append(dict(label=label,cubin_sha256=sha(cb),resources=resources,**meta));save()
 def compare(a,b):
  bits=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
  af=a.contiguous();bf=b.contiguous();finite=bool(torch.isfinite(af).all()and torch.isfinite(bf).all());neq=af.view(bits)!=bf.view(bits)
  return dict(bitwise=not bool(neq.any()),different=int(neq.sum()),elements=af.numel(),finite=finite,max_abs=float((af.float()-bf.float()).abs().max())if af.numel() else 0.)
 def capture(fn):
  for _ in range(2):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 class Compiler(NVRTCCompiler):
  @classmethod
  def get_flags(cls,*a,**kw):return super().get_flags(*a,**kw)+['--ptxas-options=-v']
 class Kernel(HummingKernel):
  def __new__(cls,*a,**kw):obj=object.__new__(cls);obj.inited=False;obj.cubin_loaded=False;return obj
  @staticmethod
  def _get_compiler():return Compiler
 base=json.loads((P/'FOUR_BANK_REFERENCE_CONFIG.json').read_text())['jobs'][0]['configuration']
 for k in [256,4096]:
  free,total=torch.cuda.mem_get_info();assert free>640*1024**2,'512MiB reserve +128MiB case headroom'
  r.setdefault('per_case_memory_preflight',[]).append(dict(k=k,free=free,allocated=torch.cuda.memory_allocated(),reserve=512*1024**2));save()
  n=256;m=24;capacity=m*8
  # Logical signed code cycles, eachK16 group gets independently varying signed finite E4M3 codes.
  logical_codes=(torch.arange(n*k,device='cuda').reshape(n,k)*7+torch.arange(n,device='cuda')[:,None])%16
  packed=(logical_codes[:,::2]|(logical_codes[:,1::2]<<4)).to(torch.uint8).unsqueeze(0).contiguous()
  finite_codes=torch.tensor([v for v in range(256)if(v&127)!=127],device='cuda',dtype=torch.uint8)
  si=(torch.arange(n*(k//16),device='cuda').reshape(n,k//16)*19+torch.arange(n,device='cuda')[:,None]*31)%254
  sf=finite_codes[si].view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
  layer=torch.nn.Module();layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(sf.clone(),requires_grad=False))
  schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
  HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=1,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True)
  # This is the only candidate packed copy; standard pack retained solely as reference fixture.
  word=layer.weight.to(torch.int64)&0xffffffff
  canonical=((word&0xf00ff00f)|((word&0x00f000f0)<<4)|((word&0x0f000f00)>>4)).to(torch.int32).contiguous();del word
  q=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device='cuda')
  expected_operand=(q[logical_codes]/4*sf[0].float().repeat_interleave(16,dim=1)).to(torch.bfloat16)
  decoded=torch.empty_like(expected_operand)
  ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32);slots=torch.full((24,),-1,device='cuda',dtype=torch.int32);x=torch.zeros(m,k,device='cuda',dtype=torch.bfloat16);locks=torch.zeros(2048,device='cuda',dtype=torch.int32)
  for banks in [4]:
   probe=Probe(n,k,banks);recordkernel(probe,'probe_k%d_b%d'%(k,banks),kind='swap_and_standard',banks=banks)
   probe.unpack(canonical,layer.weight_scale,decoded);torch.cuda.synchronize();check=compare(decoded,expected_operand);r['operand_checks'].append(dict(k=k,banks=banks,**check));save();assert check['bitwise']and check['finite'],check
   c=dict(base,shape_n=n,shape_k=k,num_experts=1,block_shape=[16,128,64*banks],warp_shape=[16,64,64]);native=Kernel(**c);table=torch.tensor([0,1<<30,native.kernel_id,0],dtype=torch.int64);recordkernel(native,'pinned_k%d_b%d'%(k,banks),kind='pinned_standard_MMA',configuration=c)
   out={name:torch.full((capacity,n),37.,device='cuda',dtype=torch.bfloat16)for name in ['swap_ab','standard_mma','pinned','wrong_adjacent','wrong_left_fold']};raw={name:torch.full((capacity,n),37.,device='cuda',dtype=torch.float32)for name in ['swap_ab','standard_mma','wrong_adjacent','wrong_left_fold']}
   assert len(set(t.data_ptr()for t in list(out.values())+list(raw.values())))==9
   def pinned():
    align=moe_align_block_size(ids,16,1,ignore_invalid_expert=True)
    return ops.launch_kernel(configs=table,inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=out['pinned'],sorted_ids=align[0],expert_ids=align[1],num_tokens_padded=align[2],locks=locks,top_k=8,valid_shape_m=capacity)
   graphs={name:capture(lambda name=name:probe.launch(name,x,canonical,layer.weight_scale,slots,out[name],raw[name]))for name in raw};graphs['pinned']=capture(pinned)
   for generation,count in enumerate([0,1,7,8,9,15,16,17,23,24,9,0,8,24]):
    rng=random.Random(241+generation);tokens=list(range(m));rng.shuffle(tokens);active_tokens=sorted(tokens[:count]);validslots=[t*8+(t+generation)%8 for t in active_tokens];assert len(validslots)==len(set(validslots))==count
    hostslots=validslots+[-1]*(24-count);assert all(s>=0 for s in hostslots[:count])and all(s<0 for s in hostslots[count:]);slots.copy_(torch.tensor(hostslots,device='cuda',dtype=torch.int32));hostids=[-1]*capacity
    for s in validslots:hostids[s]=0
    ids.copy_(torch.tensor(hostids,device='cuda',dtype=torch.int32).reshape(m,8))
    # Cancellation / subunit inputs avoid overflow despite testing maximum signed FP8 scale.
    torch.manual_seed(819+generation)
    if generation%3==0:x.copy_((torch.arange(m*k,device='cuda').reshape(m,k)%2*2-1).float()/128)
    elif generation%3==1:x.copy_(torch.randn_like(x)/128)
    else:x.copy_(((torch.arange(k,device='cuda')%16-8)[None,:].expand(m,-1)/1024).to(torch.bfloat16))
    engineered=generation>=12
    if engineered:
     # q4(code6),scale1 makes quarterweight1. One nonzero product in each64-widebank.
     canonical.fill_(0x66666666);layer.weight.fill_(0x66666666);layer.weight_scale.fill_(1.)
     x.zero_()
     partials=[2**24,1,-2**24,1] if generation==12 else [2**24,-2**24,1,1]
     for bank,value in enumerate(partials):x[:,64*bank]=value
    for t in list(out.values())+list(raw.values()):t.fill_(37.)
    for g in graphs.values():g.replay()
    torch.cuda.synchronize();active=ids.flatten()==0
    a=compare(out['swap_ab'][active],out['standard_mma'][active]);b=compare(out['standard_mma'][active],out['pinned'][active]);f=compare(raw['swap_ab'][active],raw['standard_mma'][active]);sentinel=all(bool((t[~active]==37).all())for t in list(out.values())+list(raw.values()))
    post=torch.tensor([.125,.75,1.375,2.][generation%4],device='cuda',dtype=torch.bfloat16)
    post_swap=compare(out['swap_ab'][active]*post,out['standard_mma'][active]*post);post_native=compare(out['standard_mma'][active]*post,out['pinned'][active]*post)
    engineered_checks=None
    if engineered:
     expected=2. if generation==12 else 1.;wrong=1. if generation==12 else 2.
     engineered_checks=dict(expected_raw=expected,expected_bf16_x4=expected*4,raw_known=all(bool((raw[name][active]==expected).all())for name in ('swap_ab','standard_mma')),post_known=all(bool((out[name][active]==expected*4).all())for name in ('swap_ab','standard_mma','pinned')),wrong_adjacent_reproduced=bool((raw['wrong_adjacent'][active]==wrong).all()),wrong_left_fold_reproduced=bool((raw['wrong_left_fold'][active]==wrong).all()))
    row=dict(k=k,banks=banks,generation=generation,count=count,swap_vs_standard=a,standard_vs_pinned=b,fp32_swap_vs_standard=f,external_bf16_postscale_swap=post_swap,external_bf16_postscale_pinned=post_native,sentinel=sentinel,engineered_checks=engineered_checks);r['rows'].append(row);save()
    if engineered:assert all(engineered_checks[z]for z in ('raw_known','post_known','wrong_adjacent_reproduced','wrong_left_fold_reproduced')),row
    assert a['bitwise']and f['bitwise']and post_swap['bitwise'],row
    assert a['finite']and b['finite']and f['finite']and sentinel,row
    # Standard control must reproduce pinned native arithmetic. Swap differences are recorded, not relabeled as precision lowering.
    assert b['bitwise']and post_native['bitwise']and post_swap['finite']and post_native['finite'],row
   del graphs,native,probe,out,raw
  del canonical,layer,decoded,expected_operand,logical_codes,packed,sf,x,ids,slots,locks
 r.update(status='COMPLETE_FOUR_BANK_ARITHMETIC_GATE',sources_unchanged=all(sha(f)==h for f,h in r['sources'].items()),peak_allocated=torch.cuda.max_memory_allocated(),swap_bf16_all_bitwise=all(x['swap_vs_standard']['bitwise']for x in r['rows']),swap_fp32_all_bitwise=all(x['fp32_swap_vs_standard']['bitwise']for x in r['rows']));save();assert r['sources_unchanged'];print(json.dumps({key:r[key]for key in ['status','swap_bf16_all_bitwise','swap_fp32_all_bitwise','peak_allocated']}),flush=True)
if __name__=='__main__':main()
