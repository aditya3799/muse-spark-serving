"""Actual backend buffer-method AST; structural guards only, no CUDA inference."""
import ast,json
from pathlib import Path
from types import SimpleNamespace as S
here=Path(__file__).resolve().parent
source=here/'integration/sources/python/sglang/srt/layers/attention/flashattention_backend.py'
tree=ast.parse(source.read_text());fn=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='get_fused_kv_write_buffers')
class Buffer:
 def __init__(self,shape=(173,2,64),strides=(128,64,1)):self.shape=shape;self.strides=strides
 def stride(self):return self.strides
state=S(cp=False)
class Loc:
 @staticmethod
 def for_layer(batch,layer,*,swa_loc):return S(loc='full_destination',swa_loc=swa_loc)
env=dict(torch=S(bfloat16='bf16',float8_e4m3fn='fp8'),KVWriteLoc=Loc,is_cp_active=lambda b:state.cp)
exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_backend_method','exec'),env)
call=env[fn.name];records=[]
def reset():
 state.cp=False;buffers=(Buffer(),Buffer())
 backend=S(kv_cache_dtype='fp8',fa_impl_ver=3,use_mla=False,kv_index_translator=S(is_translating=False),_kv_shard_pool=None,
  fa_skip_kv_cache=False,use_sliding_window_kv_pool=True,forward_metadata=S(swa_out_cache_loc='swa_destination'),token_to_kv_pool=S(get_kv_buffer=lambda layer:buffers))
 layer=S(k_scale=None,v_scale=None,head_dim=64,v_head_dim=64,is_cross_attention=False,tp_k_head_num=2,tp_v_head_num=2,layer_id=3,sliding_window_size=2047)
 return backend,layer,S(dcp_kv_mask=None),buffers
def check(name,change,expected=None,enabled=True):
 b,l,f,buffers=reset();change(b,l,f,buffers);out=call(b,l,f,allow_fp8_prologue=enabled)
 actual=None if out is None else out[-1];assert actual==expected,(name,actual)
 records.append(dict(case=name,result=actual))
check('full_pool_authority',lambda b,l,f,x:setattr(l,'sliding_window_size',-1),'full_destination')
check('separate_swa_pool_authority',lambda b,l,f,x:None,'swa_destination')
check('shared_pool_authority',lambda b,l,f,x:setattr(b,'use_sliding_window_kv_pool',False),'full_destination')
check('fp8_default_no_change',lambda b,l,f,x:None,enabled=False)
for name,change in [
 ('scale',lambda b,l,f,x:setattr(l,'k_scale',1)),('v_scale',lambda b,l,f,x:setattr(l,'v_scale',2)),
 ('asymmetric_vdim',lambda b,l,f,x:setattr(l,'v_head_dim',32)),('asymmetric_heads',lambda b,l,f,x:setattr(l,'tp_v_head_num',1)),
 ('mla',lambda b,l,f,x:setattr(b,'use_mla',True)),('cross',lambda b,l,f,x:setattr(l,'is_cross_attention',True)),
 ('translate',lambda b,l,f,x:setattr(b.kv_index_translator,'is_translating',True)),('shard',lambda b,l,f,x:setattr(b,'_kv_shard_pool','shard')),
 ('skip_cache',lambda b,l,f,x:setattr(b,'fa_skip_kv_cache',True)),('cp',lambda b,l,f,x:setattr(state,'cp',True)),
 ('dcp',lambda b,l,f,x:setattr(f,'dcp_kv_mask','mask')),('missing_swa',lambda b,l,f,x:setattr(b.forward_metadata,'swa_out_cache_loc',None)),
 ('buffer_shape',lambda b,l,f,x:setattr(x[0],'shape',(173,1,128))),('buffer_stride',lambda b,l,f,x:setattr(x[1],'strides',(256,128,2))),
 ('wrong_version',lambda b,l,f,x:setattr(b,'fa_impl_ver',4))]:check(name,change)
check('existing_bf16_default',lambda b,l,f,x:setattr(b,'kv_cache_dtype','bf16'),'swa_destination',enabled=False)
(here/'backend_guard_cpu.json').write_text(json.dumps(dict(scope=__doc__,passed=True,cases=records),indent=2))
print('PASS',len(records),'backend ownership/guard controls')
