"""Actual native/patched deferred_argmax AST dispatch with real CPU fallback.

The local softcap is a CPU surrogate and the group copies a virtual peer
payload. No GPU/NCCL/server startup or performance is tested here; root's
separately hash-bound GPU helper receipts cover the optimized kernel itself.
"""
import ast,hashlib,importlib.util,json,os,sys,types
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)
here=Path(__file__).resolve().parent;folder=here/'target_selector_integration'
helper_path=folder/'sources/python/sglang/kernels/ops/sampling/dflash_target_select.py'
spec=importlib.util.spec_from_file_location('sglang.kernels.ops.sampling.dflash_target_select',helper_path)
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
calls=[];real=helper.fused_select
def tracked(pairs):
 calls.append(dict(shape=list(pairs.shape),dtype=str(pairs.dtype),device=str(pairs.device),contiguous=pairs.is_contiguous()))
 return real(pairs)
helper.fused_select=tracked
class Group:
 def __init__(self,pairs):self.world_size=pairs.shape[0];self.rank_in_group=0;self.pairs=pairs;self.collectives=0
 def all_gather_into_tensor(self,out,inp):
  self.collectives+=1;out.copy_(self.pairs.reshape(out.shape))
current=[None];producer_calls=[]
def local(logits,cap,width,offset):
 producer_calls.append((tuple(logits.shape),cap,width,offset))
 # Real CPU arithmetic surrogate; native kernel is intentionally not called.
 if logits.shape[0]==0:return torch.empty((0,2),dtype=logits.dtype)
 scores=cap*torch.tanh(logits[:,:width]/cap);value,index=scores.max(-1)
 return torch.stack((value,(index+offset).to(value.dtype)),-1)
env=dict(torch=torch,os=os,softcap_row_argmax=local,get_parallel=lambda:types.SimpleNamespace(tp_group=current[0]))
methods=[]
for path in (here/'dflash_sources/logits_processor.py',folder/'sources/python/sglang/srt/layers/logits_processor.py'):
 tree=ast.parse(path.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LogitsProcessor');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='deferred_argmax')
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env);methods.append(env['deferred_argmax'])
lp=types.SimpleNamespace(vocab_size=202048,final_logit_softcapping=20.0)
generator=torch.Generator().manual_seed(791357);rows=[]
for tp in (1,2,8,16,17):
 for n in (0,1,7,32,257):
  for pattern in ('random','first_tie','nan_rank','signed_zero','negative_inf'):
   pairs=torch.empty((tp,n,2),dtype=torch.float32);pairs[...,0]=torch.randn((tp,n),generator=generator)
   pairs[...,1]=torch.arange(tp*n).view(tp,n)+5
   if pattern=='first_tie':pairs[...,0].fill_(1)
   if pattern=='nan_rank' and n:pairs[-1,:,0]=float('nan')
   if pattern=='signed_zero':pairs[...,0].zero_();pairs[::2,:,0]=-0.0
   if pattern=='negative_inf':pairs[...,0].fill_(-float('inf'))
   logits=torch.randn((n,19),generator=generator);group=Group(pairs);current[0]=group
   calls.clear();producer_calls.clear();reference=methods[0](lp,logits);native_producer=producer_calls[-1]
   for enabled in ('0','1'):
    os.environ['SGLANG_EXPERIMENTAL_DFLASH_TARGET_SELECT']=enabled
    before=group.collectives;calls.clear();producer_calls.clear();actual=methods[1](lp,logits)
    assert torch.equal(actual,reference)
    assert group.collectives==before+1 and producer_calls==[native_producer]
    assert len(calls)==int(enabled)
    rows.append(dict(tp=tp,n=n,pattern=pattern,enabled=enabled=='1',exact=True,native_producer_unchanged=True,collectives=1,
      logical_cuda_geometry_supported=(tp<=16 and n>0),actual_route='real_CPU_native_fallback' if enabled=='1' else 'original_inline_postops'))
fallback=[]
for name,pairs in [('noncontiguous',torch.randn((8,7,4),generator=generator)[...,::2]),('fp16',torch.randn((8,7,2),generator=generator).half()),('bf16',torch.randn((8,7,2),generator=generator).bfloat16()),('empty_tokens',torch.empty((8,0,2))),('tp17',torch.randn((17,7,2),generator=generator))]:
 assert torch.equal(real(pairs),helper.native_select(pairs));fallback.append(name)
out=dict(scope=__doc__,passed=True,helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
 candidate_sha256=hashlib.sha256((folder/'candidate.diff').read_bytes()).hexdigest(),cases=rows,direct_fallback_controls=fallback,
 gpu_used=False,real_collective=False,performance_measured=False)
(here/'target_selector_cpu.json').write_text(json.dumps(out,indent=2));print('PASS',len(rows),'actual method dispatch cases +',len(fallback),'fallback controls')
