"""CPU-only validation contracts for the exact two compiled fused W13 jobs."""
from pathlib import Path
import json
from virtual_k_gate_contract import require_h200
P=Path(__file__).resolve().parent

def approved_job(job_id,configuration=None):
 jobs=json.loads((P/'w13_fused_jobs.json').read_text(encoding='utf-8-sig'))['jobs']
 hits=[j for j in jobs if j['job_id']==job_id]
 if len(hits)!=1:raise ValueError('Only the two audited SM90 jobs are supported')
 j=hits[0];c=j['configuration']
 if configuration is not None and configuration!=c:raise ValueError('Configuration differs from compiled whitelist')
 if not c.get('use_cp_async') or c.get('use_stream_k') or c.get('use_f16_accum') or c.get('use_warp_spec'):raise ValueError('Unsupported pipeline')
 if any(v for k,v in c.items() if k.startswith('use_tma')):raise ValueError('TMA descriptors are not implemented')
 if c['num_experts']!=32 or c['shape_n']!=8192 or c['shape_k']!=4096:raise ValueError('Unsupported model geometry')
 return j

def validate_tensors(layer,x,out,ids,aligned,c):
 m=x.shape[0];h=c['shape_k'];e=c['num_experts'];n=c['shape_n']
 if tuple(x.shape)!=(m,h) or tuple(out.shape)!=(m*8,n//2) or tuple(ids.shape)!=(m,8):raise ValueError('Input/output/route shape mismatch')
 floats=(x,out,layer.w13_post_scale)
 if any(str(t.dtype)!='torch.bfloat16' for t in floats):raise ValueError('BF16 inputs, output and scale required')
 ints=(ids,*aligned,layer.locks)
 if len(aligned)!=3 or any(str(t.dtype)!='torch.int32' for t in ints):raise ValueError('Int32 routing and locks required')
 tensors=(*floats,*ints,layer.w13_weight,layer.w13_weight_scale)
 if any(not t.is_contiguous() for t in tensors):raise ValueError('Contiguous tensors required')
 if any(t.device!=x.device for t in tensors):raise ValueError('Single-device tensors required')
 if tuple(layer.w13_post_scale.shape)!=(e,):raise ValueError('Expert scale shape mismatch')
 if str(layer.w13_weight.dtype)!='torch.int32' or layer.w13_weight.numel()!=e*n*h//8:raise ValueError('Packed weight contract mismatch')
 if str(layer.w13_weight_scale.dtype)!='torch.float8_e4m3fn' or layer.w13_weight_scale.numel()!=e*n*h//16:raise ValueError('Packed FP8 scale contract mismatch')
 if len(aligned[0].shape)!=1 or len(aligned[1].shape)!=1 or tuple(aligned[2].shape)!=(1,):raise ValueError('Alignment buffer rank mismatch')
 if aligned[0].numel()!=aligned[1].numel()*c['block_shape'][0]:raise ValueError('Alignment capacity mismatch')
 if out.data_ptr()==x.data_ptr():raise ValueError('Output must not alias activation input')
