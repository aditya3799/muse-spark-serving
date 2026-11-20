"""Actual AST metadata ownership controls; no CUDA, SGLang import or model mutation."""
import ast, json, hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
class Tensor:
    def __init__(self, values, shape=None, name=''):
        self.values=list(values); self.shape=shape or (len(self.values),);self.name=name
    def __getitem__(self,key):return Tensor(self.values[key],name=self.name+'/slice')
    def stride(self):return (128,64,1)
paths={
 'scope':ROOT/'sglang/python/sglang/srt/layers/attention/graph_utils.py',
 'loc':ROOT/'sglang/python/sglang/srt/mem_cache/memory_pool.py',
 'backend':ROOT/'research_2026-11-19_round2/decode/integration/sources/python/sglang/srt/layers/attention/flashattention_backend.py'}
ns=dict(torch=NS(Tensor=Tensor,bfloat16='bf16',float8_e4m3fn='fp8'),Optional=Optional,dataclass=dataclass,contextmanager=contextmanager,is_cp_active=lambda fb:fb.cp)
def extract(key,name):
 tree=ast.parse(paths[key].read_text()); node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),str(paths[key]),'exec'),ns)
extract('scope','attention_input_scope');extract('loc','KVWriteLoc');extract('backend','get_fused_kv_write_buffers')
records=[]
for sliding in (False,True):
 for throws in (False,True):
  for generation in (0,1):
   full=Tensor(range(10+generation*100,18+generation*100));swa=Tensor(range(50+generation*100,58+generation*100))
   positions=Tensor(range(8));original_output=object();output=Tensor(range(8))
   fb=NS(out_cache_loc=full,positions=positions,_attn_output=original_output,out_cache_loc_is_physical=True,cp=False,dcp_kv_mask=None)
   layer=NS(k_scale=None,v_scale=None,head_dim=64,v_head_dim=64,tp_k_head_num=2,tp_v_head_num=2,is_cross_attention=False,sliding_window_size=2047 if sliding else -1,layer_id=0)
   buffers=(Tensor([],shape=(1000,2,64)),Tensor([],shape=(1000,2,64)))
   backend=NS(kv_cache_dtype='fp8',fa_impl_ver=3,use_mla=False,kv_index_translator=NS(is_translating=False),_kv_shard_pool=None,fa_skip_kv_cache=False,use_sliding_window_kv_pool=True,forward_metadata=NS(swa_out_cache_loc=swa),token_to_kv_pool=NS(get_kv_buffer=lambda _:buffers))
   try:
    with ns['attention_input_scope'](fb,output,3,3,{},cache_tokens=3):
     got=ns['get_fused_kv_write_buffers'](backend,layer,fb,allow_fp8_prologue=True)
     assert got[:2]==buffers
     assert got[2].values==(swa if sliding else full).values[:3]
     assert len(got[2].values)==3 and fb.positions.values==[0,1,2]
     assert fb._attn_output.values==[0,1,2]
     if throws:raise RuntimeError('injected_after_resolution')
   except RuntimeError as exc:
    assert throws and str(exc)=='injected_after_resolution'
   assert fb.out_cache_loc is full and fb.positions is positions and fb._attn_output is original_output
   records.append(dict(sliding=sliding,exception=throws,generation=generation,passed=True))
# Preserve the frozen buffer API's fail-closed opt-in and scale/translation guards.
negative=[]
for field,value in [('allow_fp8_prologue',False),('fa_impl_ver',4),('use_mla',True),('k_scale',1.),('is_cross_attention',True),('fa_skip_kv_cache',True),('dcp_kv_mask',object()),('cp',True)]:
 target=layer if field in ('k_scale','is_cross_attention') else fb if field in ('dcp_kv_mask','cp') else backend
 if field=='allow_fp8_prologue':got=ns['get_fused_kv_write_buffers'](backend,layer,fb)
 else:
  previous=getattr(target,field);setattr(target,field,value)
  got=ns['get_fused_kv_write_buffers'](backend,layer,fb,allow_fp8_prologue=True);setattr(target,field,previous)
 assert got is None;negative.append(field)
budget=[]
for m,p in [(56,64),(454,512),(753,1024),(1024,1024),(2003,2048),(2016,2048),(2016,4096),(3012,4096),(3641,4096),(16383,16384)]:
 existing=m*8192+262144
 estimate=existing+2304*(p-m)
 budget.append(dict(real=m,padded=p,original_decoder_estimate=existing,padded_reference_estimate=estimate,admit=estimate<=16*1024**2))
doc=dict(scope=__doc__,passed=True,actual_source_cases=records,negative_guards=negative,budget_cases=budget,
 sources={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
 limitations=['Tensor slicing surrogate only; no native CUDA or BCG execution.','No serving activation or replacement of frozen runtime.','Budget is conservative source accounting; allocator/graph/runtime measurement required.'])
(HERE/'prefill_bcg_source_controls.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(dict(passed=True,metadata_cases=len(records),negative_guards=len(negative),budget_cases=len(budget))))
