"""CPU-only contracts for future corrected Hopper gates."""
from pathlib import Path
import hashlib,json,random
P=Path(__file__).resolve().parent
EXPECTED_MANIFEST_SHA256='3406cb417f1579f887015ec355bb3ac60ebec27f3994ed229585b54e3b590fa2'

def require_h200(major,minor,name):
 if (major,minor)!=(9,0) or 'H200' not in name.upper():raise RuntimeError('Requires H200 SM90; SM86/other models refused')
def validate_config(c):
 expected={'a_dtype':'bfloat16','b_dtype':'float4e2m1','bs_dtype':'float8e4m3','c_dtype':'bfloat16','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','mma_type':'wgmma','gemm_type':'indexed','pad_shape_n':0,'pad_shape_k':0}
 if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Unqualified arithmetic/layout')
 if any(v for k,v in c.items() if k.startswith('use_tma')):raise ValueError('TMA descriptors unsupported')
 if any(c.get(k,False) for k in ('use_warp_spec','use_stream_k','use_f16_accum','use_packed_k_layout','has_bias','has_zero_point','use_fused_e8m0_scale','use_int_weight_scale')):raise ValueError('Unqualified feature')
 if c.get('input_scale_group_size',0) or c.get('num_write_splits',1)!=1 or not c.get('use_cp_async',False):raise ValueError('Unsupported scale/split/producer')
 if c['shape_k']%c['block_shape'][2] or c['num_stages'] not in (3,4):raise ValueError('K tail/stages unsupported')
 if any(c.get(k,1)!=1 for k in ('multi_cast_size_a','multi_cast_size_b')):raise ValueError('Cluster unsupported')
def load_manifest():
 path=P/'H200_CORRECTED_GATE_MANIFEST.json';manifest=json.loads(path.read_text(encoding='utf-8-sig'))
 if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_MANIFEST_SHA256:raise ValueError('Gate manifest changed')
 V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor'
 for name,digest in manifest['pinned_vendor_hashes'].items():
  if hashlib.sha256((V/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned vendor changed: '+name)
 for name,digest in manifest['research_source_hashes'].items():
  if hashlib.sha256((P/name).read_bytes()).hexdigest()!=digest:raise ValueError('Research source changed: '+name)
 for pair in manifest['pairs']:
  for side in ('baseline','candidate'):
   x=pair[side];validate_config(x['configuration'])
   for kind in ('source','cubin'):
    path=(P/x[kind+'_path']).resolve()
    if (P/'h200_gate_assets').resolve() not in path.parents:raise ValueError('Artifact path escape')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=x[kind+'_sha256']:raise ValueError('Artifact changed')
 return manifest

def local_routes(m,experts,pattern,seed):
 rng=random.Random(seed);ids=[]
 for token in range(m):
  if pattern=='uniform':row=rng.sample(range(experts*8),8)
  elif pattern=='one_hot':row=[0]+rng.sample(range(experts,experts*8),7)
  elif pattern=='eight_hot':row=rng.sample(range(8),8)
  elif pattern=='mixed':row=rng.sample(range(8),8) if token<m//2 else rng.sample(range(experts,experts*8),8)
  elif pattern=='all_remote':row=rng.sample(range(experts,experts*8),8)
  else:raise ValueError(pattern)
  assert len(set(row))==8;ids.extend(x if x<experts else -1 for x in row)
 return ids

def aligned_routes(ids,experts,bm):
 slots=len(ids);cap=((slots+experts*(bm-1)+bm-1)//bm)*bm;sorted_ids=[slots]*cap;expert_ids=[-1]*(cap//bm);off=0
 for expert in range(experts):
  local=[i for i,x in enumerate(ids) if x==expert];pad=((len(local)+bm-1)//bm)*bm;sorted_ids[off:off+len(local)]=local;expert_ids[off//bm:(off+pad)//bm]=[expert]*(pad//bm);off+=pad
 assert off<=cap
 return sorted_ids,expert_ids,off

def validate_tensors(x,weight,scale,out,sorted_ids,expert_ids,padded,c,slots,topk):
 tensors=(x,weight,scale,out,sorted_ids,expert_ids,padded)
 if any(not t.is_cuda or not t.is_contiguous() or t.device!=x.device for t in tensors):raise ValueError('All tensors must be contiguous on one CUDA device')
 expected=((x,'torch.bfloat16'),(weight,'torch.int32'),(scale,'torch.float8_e4m3fn'),(out,'torch.bfloat16'),(sorted_ids,'torch.int32'),(expert_ids,'torch.int32'),(padded,'torch.int32'))
 if any(str(t.dtype)!=dtype for t,dtype in expected):raise ValueError('Dtype mismatch')
 e,n,k=c['num_experts'],c['shape_n'],c['shape_k'];bm=c['block_shape'][0]
 if tuple(x.shape)!=(slots//topk,k) or tuple(out.shape)!=(slots,n):raise ValueError('Activation/output shape mismatch')
 if weight.numel()*4!=e*n*k//2 or scale.numel()!=e*n*k//16:raise ValueError('Canonical packed weight/scale extent mismatch')
 if sorted_ids.ndim!=1 or expert_ids.ndim!=1 or sorted_ids.numel()!=expert_ids.numel()*bm or padded.numel()!=1:raise ValueError('Route-buffer extent mismatch')
