from pathlib import Path
import sys,json,copy,math,hashlib
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');sys.path.insert(0,str(P));from h200_corrected_contract import *
class Tensor:
 def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype;self.is_cuda=True;self.device='cuda:0';self.contiguous=True;self.ndim=len(shape)
 def is_contiguous(self):return self.contiguous
 def numel(self):return math.prod(self.shape)
checks=0
for pair in load_manifest()['pairs']:
 for side in ('baseline','candidate'):
  c=pair[side]['configuration'];slots=pair['tokens']*8;topk=8 if pair['stage']=='w13' else 1;e,n,k=c['num_experts'],c['shape_n'],c['shape_k'];bm=c['block_shape'][0];si,ei,np=aligned_routes(local_routes(pair['tokens'],e,'uniform',4),e,bm)
  ts=[Tensor((slots//topk,k),'torch.bfloat16'),Tensor((e*n*k//8,),'torch.int32'),Tensor((e*n*k//16,),'torch.float8_e4m3fn'),Tensor((slots,n),'torch.bfloat16'),Tensor((len(si),),'torch.int32'),Tensor((len(ei),),'torch.int32'),Tensor((1,),'torch.int32')]
  validate_tensors(*ts,c,slots,topk);checks+=1
  for idx in range(7):
   for field,value in [('dtype','torch.float32'),('is_cuda',False),('contiguous',False),('device','cuda:1')]:
    bad=copy.deepcopy(ts);setattr(bad[idx],field,value)
    try:validate_tensors(*bad,c,slots,topk)
    except ValueError:checks+=1
    else:raise AssertionError((idx,field))
  for idx in range(7):
   bad=copy.deepcopy(ts);bad[idx].shape=(1,)
   try:validate_tensors(*bad,c,slots,topk)
   except ValueError:checks+=1
   else:
    assert idx==6
r={'status':'CPU_METADATA_CONTRACT_PASS_NO_TORCH_OR_CUDA','checks':checks,'scope':'Fake-tensor shape/dtype/contiguity/device/extent preflight only, not kernel memory validation','gate_sha256':hashlib.sha256((P/'h200_corrected_gate.py').read_bytes()).hexdigest(),'contract_sha256':hashlib.sha256((P/'h200_corrected_contract.py').read_bytes()).hexdigest()};(P/'H200_TENSOR_GUARD_CPU_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
