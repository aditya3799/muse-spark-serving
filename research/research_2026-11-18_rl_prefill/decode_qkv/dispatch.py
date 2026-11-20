"""Run the scored and candidate Muse attention forward methods around the owned write and prove cumulative apply."""
import ast,hashlib,json,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace as S
import torch
H=Path(__file__).resolve().parent;ROOT=H.parents[1];REPO=ROOT/'sglang';PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
def extract(path,cls,name,scope):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(f for n in tree.body if getattr(n,'name',None)==cls for f in n.body if getattr(f,'name',None)==name)
    scope=dict(torch=torch,**scope);exec(compile(ast.Module(ast.parse('from __future__ import annotations').body+[node],[]),str(path),'exec'),scope);return scope[name]
class Gate:
    def __init__(self,trace,eligible):self.trace=trace;self.eligible=eligible
    def can_fuse_fa3(self,*_):return self.eligible
    def __call__(self,qkv,*_,**kwargs):self.trace.append('original_norm');return qkv[:,:1024],qkv[:,1024:1152]
    def _forward_triton(self,qkv,*_):self.trace.append('fused_fp8_append');return qkv[:,:1024].to(torch.float8_e4m3fn),qkv[:,1024:1152]
class Radix:
    k_scale=None;v_scale=None
    def __init__(self,trace):self.trace=trace
    def __call__(self,q,k,v,fb,save_kv_cache):
        self.trace.append(('radix',save_kv_cache,str(q.dtype)));return q.to(torch.bfloat16)
def go(arm,mode,one,unified,eligible,valid,rope):
    trace=[];T=8 if mode=='verify' else 1;h=torch.ones(T,2304,dtype=torch.bfloat16);backend=S(muse_fp8_write_buffers=lambda *args:(None,None,torch.zeros(T if valid else 0),None))
    mem=S(enable_unified_memory=unified)
    ns=dict(get_memory=lambda:mem,get_attn_backend=lambda:backend,get_token_to_kv_pool=lambda:None,TRTLLMHAAttnBackend=type('OtherBackend',(),{}),is_cp_active=lambda _:False)
    p=H/(('parent_' if arm=='baseline' else '')+'muse_spark.py')
    forward=extract(p,'MuseSpark1xAttention','forward',ns);getter=extract(p,'MuseSpark1xAttention','_fused_kv_store',ns)
    op=S(q_size=1024,kv_size=128,use_o_gate=True,qkv_proj=lambda x:(x,None),attn=Radix(trace),_fused_qk_norm=Gate(trace,eligible),rotary_emb=S(cos_sin_cache=object()) if rope else None,_fused_o_norm_gate=lambda x,g:x,o_proj=lambda x:(x,None))
    op._fused_kv_store=lambda fb:getter(op,fb)
    fb=S(forward_mode=S(is_decode=lambda:mode=='decode',is_target_verify=lambda:mode=='verify'),dcp_kv_mask=None)
    if not one: backend.muse_fp8_write_buffers=lambda *args:None
    out=forward(op,torch.arange(T),h,fb)
    expected=(arm=='candidate' and mode in ('decode','verify') and not unified and eligible and valid and one)
    assert trace[0]==('fused_fp8_append' if expected else 'original_norm') and trace[1][1]==(not expected)
    assert out.shape==(T,1024)
    return dict(arm=arm,mode=mode,eligible=expected,trace=trace)
cases=[]
for a in ('baseline','candidate'):
 for mode in ('decode','verify','prefill'):
  for enabled,unified,eligible,valid in ((True,False,True,True),(False,False,True,True),(True,True,True,True),(True,False,False,True),(True,False,True,False)):
   for rotary in (False,True):cases.append(go(a,mode,enabled,unified,eligible,valid,rotary))
with tempfile.TemporaryDirectory(dir=H) as x:
    env=dict(os.environ,GIT_INDEX_FILE=str(Path(x)/'temp.index'))
    def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],env=env)
    git('read-tree',PIN);git('apply','--cached','--check',str(H/'candidate.diff'))
    git('apply','--cached',str(H/'candidate.diff'))
    for path in ('python/sglang/srt/models/muse_spark.py','python/sglang/srt/layers/attention/flashattention_backend.py','python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py'):
        assert git('show',':'+path)==(H/Path(path).name).read_bytes()
result=dict(cases=cases,n=len(cases),cumulative_apply_check=True,scope='Actual source-defined target model dispatch/FA3 get_write eligibility under stated flags. Traced QK and radix methods in this dispatch-only fixture; real entry kernels independently tested on RTX. H200 startup log contains configuration, not runtime per-layer counters.')
(H/'dispatch.json').write_text(json.dumps(result,indent=2));print('qualified',len(cases))

