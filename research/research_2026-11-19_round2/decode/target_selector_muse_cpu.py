"""Pinned Muse subclass inheritance/FP32 policy; no CUDA or server startup."""
import ast,hashlib,json,os,types
from pathlib import Path
here=Path(__file__).resolve().parent
# Reuse only the frozen test's imports/helpers and actual-method extraction.
# Stop before its test loop and receipt write, preserving every frozen receipt.
fixture=(here/'target_selector_cpu.py').read_text()
prefix=fixture[:fixture.index('lp=types.SimpleNamespace')]
env={'__file__':str(here/'target_selector_cpu.py'),'__name__':'muse_dispatch_support'}
exec(compile(prefix,'frozen_selector_test_prefix','exec'),env)
torch=env['torch'];source=(here/'muse_logits_activation/muse_spark.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xLogitsProcessor')
rows=[];policy=[]
for base_index,method in enumerate(env['methods']):
 class Base:
  def __init__(self,config,**kwargs):
   self.vocab_size=config.vocab_size
   self.final_logit_softcapping=config.final_logit_softcapping
 Base.deferred_argmax=method
 env.update(OSSLogitsProcessor=Base,use_intel_amx_backend=lambda head:False)
 exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_muse_class','exec'),env)
 Muse=env['MuseSpark1xLogitsProcessor']
 config=types.SimpleNamespace(vocab_size=202048,final_logit_softcapping=20.0,enable_lm_head_fp32=False,enable_lm_head_output_fp32=False)
 lp=Muse(config)
 assert lp.use_fp32_output is True and Muse.deferred_argmax is method
 # CPU-only spy on actual native override: verifies requested dtype, not GEMM numerics.
 mm_calls=[]
 def mm_spy(a,b,**kwargs):mm_calls.append(kwargs);return 'fp32_output_sentinel'
 env['torch']=types.SimpleNamespace(float32=torch.float32,mm=mm_spy)
 weight=types.SimpleNamespace(dtype=torch.bfloat16,shape=(19,4),T='transposed_weight')
 hidden=types.SimpleNamespace(shape=(7,4),to=lambda dtype:'cast_hidden')
 assert lp._compute_lm_head(hidden,types.SimpleNamespace(weight=weight))=='fp32_output_sentinel'
 assert mm_calls==[dict(out_dtype=torch.float32,out=None)]
 env['torch']=torch
 policy.append(dict(base='native' if base_index==0 else 'patched',inherits_exact_method=True,fp32_mm_requested_with_generic_flags_false=True))
 for enabled in ('0','1'):
  os.environ['SGLANG_EXPERIMENTAL_DFLASH_TARGET_SELECT']=enabled
  for n in (0,1,7,32):
   pairs=torch.zeros((8,n,2));pairs[...,1]=torch.arange(8*n).view(8,n)
   if n:pairs[3,:,0]=float('nan')
   group=env['Group'](pairs);env['current'][0]=group
   env['calls'].clear();actual=lp.deferred_argmax(torch.zeros((n,19)))
   expected=env['real'](pairs)
   assert torch.equal(actual,expected) and group.collectives==1
   assert len(env['calls'])==(int(enabled) if base_index else 0)
   rows.append(dict(base=base_index,enabled=enabled,n=n,exact=True,gather_count=1))
out=dict(passed=True,scope=__doc__,cases=rows,policy=policy,gpu_used=False,
 base_constructor='minimal CPU metadata stub; actual Muse constructor, override and inherited deferred method executed',
 mm='spy only; no GEMM numerical proof',softcap='CPU surrogate',transport='virtual-peer copy',
 source_sha256=hashlib.sha256(source.encode()).hexdigest(),helper_sha256=hashlib.sha256(env['helper_path'].read_bytes()).hexdigest())
(here/'target_selector_muse_cpu.json').write_text(json.dumps(out,indent=2));print('PASS',len(rows),'Muse inherited-method dispatch cases +',len(policy),'native FP32 policy spies')
