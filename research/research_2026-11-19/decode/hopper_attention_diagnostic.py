"""Unexecuted Hopper-only differential harness for the precision-restored donor.

Requires an explicitly reviewed SM90a shared library. No automatic compilation,
remote dispatch, tuning or serving activation. Reports raw differences; cannot
grant serving eligibility or inherit the public donor's performance claims.
"""
import argparse,ctypes as C,hashlib,json,math
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
POINTERS='qkv seq_lens cache_loc k_cache v_cache page_table k_scale v_scale inv_freq positions out ws_acc ws_l ws_cnt'.split()
INTS='B Hq Hkv P max_pages window rope n_splits flat nc'.split()
class Params(C.Structure):
 _fields_=[(n,C.c_void_p) for n in POINTERS]+[(n,C.c_int) for n in INTS]+[
  ('dbg',C.c_void_p),('cs',C.c_int),('ncl',C.c_int),('ld_qkv',C.c_int),('row_bytes',C.c_int),
  ('scale_log2',C.c_float),('coop',C.c_int),('lp',C.c_int),('lrb',C.c_int),('off32',C.c_int),('ns_magic',C.c_uint)]

def load(path):
 lib=C.CDLL(str(path));lib.attention_params_size.restype=C.c_size_t
 lib.attention_param_offset.argtypes=[C.c_int];lib.attention_param_offset.restype=C.c_size_t
 assert lib.attention_params_size()==C.sizeof(Params)
 for i,(name,_) in enumerate(Params._fields_):assert lib.attention_param_offset(i)==getattr(Params,name).offset,name
 lib.attention_diagnostic.argtypes=[C.POINTER(Params),C.c_void_p];lib.attention_diagnostic.restype=C.c_int
 return lib

def error(x,y):
 delta=x.float()-y.float()
 return dict(exact=bool(torch.equal(x.view(torch.uint8),y.view(torch.uint8))),
  changed_bytes=int(torch.count_nonzero(x.view(torch.uint8)!=y.view(torch.uint8))),
  max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/y.float().norm().clamp_min(1e-30)),finite=bool(x.float().isfinite().all()))

@torch.no_grad()
def run_case(lib,prefixes,window,rope_on,splits,seed):
 from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm
 from sglang.kernels.ops.layernorm.muse_spark_norm_gate_jit import rms_norm_gate
 from sglang.kernels.ops.kvcache.kvcache import store_cache
 import sgl_kernel.flash_attn as fa
 b=len(prefixes);q_len=8;hq=16;hkv=2;d=64;width=2304
 # -8 denotes a graph-padded request whose native total cache length is0.
 assert all(p>=0 or p==-8 for p in prefixes)
 gen=torch.Generator(device='cuda').manual_seed(seed)
 cap=max(8,max(prefixes)+8);pool=b*cap+1
 table=torch.stack([torch.randperm(cap,device='cuda',generator=gen)+1+r*cap for r in range(b)]).int()
 lens=torch.tensor([p+8 for p in prefixes],device='cuda',dtype=torch.int32)
 prefix=torch.tensor(prefixes,device='cuda',dtype=torch.int32)
 loc=torch.zeros((b,8),device='cuda',dtype=torch.int64)
 positions=torch.zeros((b,8),device='cuda',dtype=torch.int64)
 for r,p in enumerate(prefixes):
  if p>=0:loc[r]=table[r,p:p+8];positions[r]=torch.arange(p,p+8,device='cuda')
  else:table[r].zero_()
 qkv=torch.randn((b*8,width),device='cuda',dtype=torch.bfloat16,generator=gen)
 initial=[torch.randn((pool,1,hkv,d),device='cuda',generator=gen).to(torch.float8_e4m3fn) for _ in range(2)]
 for tensor in initial:tensor[0].zero_()
 native_k,native_v=[x.clone() for x in initial];candidate_k,candidate_v=[x.clone() for x in initial]
 phase=torch.arange(cap,device='cuda').float()[:,None]*500000.**(-torch.arange(0,d,2,device='cuda').float()/d)
 rope=torch.cat([phase.cos(),phase.sin()],-1).contiguous()
 norm=FusedQKNorm(head_dim=d,num_q_heads=hq,num_kv_heads=hkv,eps=1e-5)
 # Frozen Muse enables fused KV stores only for TRT, not FA3. FA3 first
 # materializes BF16 Q/K, then casts Q and BF16 K/V separately to FP8.
 # This also preserves the original Triton head-row count/configuration.
 native_q,native_k_rows=norm._forward_triton(qkv,positions.flatten() if rope_on else None,rope if rope_on else None,
  kv_store=None)
 native_q=native_q.to(torch.float8_e4m3fn)
 native_v_rows=qkv[:,hq*d+hkv*d:hq*d+2*hkv*d]
 store_cache(native_k_rows.to(torch.float8_e4m3fn).view(torch.uint8),
  native_v_rows.to(torch.float8_e4m3fn).view(torch.uint8),
  native_k.view(pool,hkv*d).view(torch.uint8),native_v.view(pool,hkv*d).view(torch.uint8),
  loc.flatten(),row_bytes=hkv*d,size_limit=pool,reserved_skip_index=0)
 cu=torch.arange(0,(b+1)*8,8,device='cuda',dtype=torch.int32)
 # Muse folds qk_scale_factor into FA3's softmax scale rather than Q storage.
 scale=.125*5.4730064863875
 attn=fa.flash_attn_with_kvcache(native_q.view(b*8,hq,d),native_k,native_v,
  cache_seqlens=lens,page_table=table,cu_seqlens_q=cu,max_seqlen_q=8,causal=True,
  window_size=(window-1,0) if window else (-1,0),softmax_scale=scale,num_splits=1,ver=3)
 if isinstance(attn,tuple):attn=attn[0]
 native=rms_norm_gate(attn.reshape(b*8,hq*d).contiguous(),qkv[:,-hq*d:],head_dim=d,eps=1e-5)
 # Empty donor splits return before stores. Zero initialization makes graph
 # padding deterministic; its cost must be charged in any later benchmark.
 output=torch.zeros((b*8,hq*d),device='cuda',dtype=torch.bfloat16)
 accum=torch.zeros((b*8*hq*d,),device='cuda');denom=torch.zeros((b*8*hq,),device='cuda')
 counters=torch.zeros(32768,device='cuda',dtype=torch.int32)
 ones=torch.ones(1,device='cuda');p=Params()
 tensors=dict(qkv=qkv,seq_lens=prefix,cache_loc=loc,k_cache=candidate_k,v_cache=candidate_v,
  page_table=table,k_scale=ones,v_scale=ones,inv_freq=rope,positions=positions,out=output,
  ws_acc=accum,ws_l=denom,ws_cnt=counters)
 for name,tensor in tensors.items():setattr(p,name,tensor.data_ptr())
 for name,value in dict(B=b,Hq=hq,Hkv=hkv,P=1,max_pages=cap,window=window,rope=int(rope_on),
  n_splits=splits,flat=0,nc=1,cs=0,ncl=1,ld_qkv=width,row_bytes=128,
  scale_log2=C.c_float(scale).value*C.c_float(math.log2(math.e)).value,coop=0,lp=0,lrb=7,
  off32=int(candidate_k.numel()<2**32),ns_magic=((2**32+splits-1)//splits if splits>1 else 0)).items():setattr(p,name,value)
 def launch():
  output.zero_()
  status=lib.attention_diagnostic(C.byref(p),torch.cuda.current_stream().cuda_stream)
  if status:raise RuntimeError(('attention launch',status))
 launch();torch.cuda.synchronize()
 first=output.clone();launch();torch.cuda.synchronize()
 return dict(prefixes=prefixes,window=window,rope=rope_on,splits=splits,seed=seed,
  k=error(candidate_k,native_k),v=error(candidate_v,native_v),output=error(output,native),
  repeated_replay=error(output,first),counter_nonzero=int(torch.count_nonzero(counters)),
  qualification='DIAGNOSTIC_ONLY; compare actual native outputs before considering any serving gate')

def main(a):
 if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(9,0):raise RuntimeError('requires an explicitly allocated Hopper SM90 GPU; no local fallback')
 lib=load(a.library);rows=[]
 for prefixes in ([0],[137],[4096],[0,129,2051,-8],[0,0,0,0,0,0,0]):
  for window,rope in ((0,False),(2048,True)):
   for splits in (1,4,16):
    row=run_case(lib,prefixes,window,rope,splits,a.seed);rows.append(row);print(json.dumps(row),flush=True)
 doc=dict(scope=__doc__,device=torch.cuda.get_device_name(),seed=a.seed,results=rows,
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
  complete=True,serving_qualified=False)
 a.output.write_text(json.dumps(doc,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=4817391)
 main(parser.parse_args())
