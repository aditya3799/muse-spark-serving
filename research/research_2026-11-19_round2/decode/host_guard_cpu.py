"""Execute actual hook AST against structural mocks; no CUDA/math claim."""
import ast,json,os,sys,types
from pathlib import Path
from types import SimpleNamespace as S
here=Path(__file__).resolve().parent
state=S(capture=False,cp=False,forced=None)
class Tensor:
 def __init__(self,shape=(173,2,64),contiguous=True):self.shape=shape;self.ndim=len(shape);self.contiguous=contiguous
 def is_contiguous(self):return self.contiguous
 def view(self,*shape):return Tensor(shape,self.contiguous)
class Backend:
 fa_impl_ver=3;is_draft_runner=False;topk=1;kv_cache_dtype='fp8'
 def get_fused_kv_write_buffers(self,*a,**kw):
  assert kw==dict(allow_fp8_prologue=True);return self.buffers
class Adapter:
 def try_run(self,*a,**kw):self.args=a;self.kw=kw;return 'fused_q'
class Norm:
 is_torch_compile=False
 def auto_selected_backend(self):return 'triton'
 def backend_eligible(self,*a,**kw):return True
 def _forward_triton(self,*a,**kw):return 'q','k'
def module(name,**values):
 obj=types.ModuleType(name);obj.__dict__.update(values);sys.modules[name]=obj
module('sglang.kernels.fused_op',get_fused_op_backend=lambda:state.forced)
module('sglang.kernels.spec',KernelBackend=S(TRITON='triton'))
module('sglang.srt.layers.attention.flashattention_backend',FlashAttentionBackend=Backend)
module('sglang.srt.layers.cp.utils',is_cp_active=lambda batch:state.cp)
module('sglang.srt.model_executor.forward_context',get_attn_backend=lambda:state.backend)
module('sglang.srt.runtime_context',get_memory=lambda:state.memory,get_parallel=lambda:state.parallel)
tree=ast.parse((here/'prologue_package/host.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef))
env=dict(os=os,torch=S(float8_e4m3fn='fp8',cuda=S(is_current_stream_capturing=lambda:state.capture)),Fa3Prologue=Adapter)
exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_host_hook','exec'),env)
hook=env['maybe_fuse'];records=[]
def reset():
 state.capture=False;state.cp=False;state.forced=None
 state.backend=Backend();state.backend.buffers=(Tensor(),Tensor(),Tensor((56,)))
 state.memory=S(enable_unified_memory=False);state.parallel=S(attn_dp_enabled=False)
 norm=Norm();attention=S(_fused_qk_norm=norm,attn=S(k_scale=None,v_scale=None),rotary_emb=S(cos_sin_cache=Tensor((4096,64))),
 head_dim=64,num_heads=16,num_kv_heads=2,norm_eps=1e-5)
 batch=S(dcp_kv_mask=None,forward_mode=S(is_decode=lambda:True,is_target_verify=lambda:False))
 return attention,batch
def check(name,mutator,expected=None):
 a,b=reset();mutator(a,b);got=hook(a,'qkv','pos',b);assert got==expected,(name,got)
 records.append(dict(case=name,result=got,adapter_created=hasattr(a,'_experimental_fa3_prologue')))
os.environ['SGLANG_EXPERIMENTAL_MUSE_FA3_PROLOGUE']='0'
check('default_off',lambda a,b:None)
os.environ['SGLANG_EXPERIMENTAL_MUSE_FA3_PROLOGUE']='1'
check('eligible_decode',lambda a,b:None,'fused_q')
check('eligible_verify',lambda a,b:setattr(b,'forward_mode',S(is_decode=lambda:False,is_target_verify=lambda:True)),'fused_q')
check('prefill',lambda a,b:setattr(b,'forward_mode',S(is_decode=lambda:False,is_target_verify=lambda:False)))
check('forced_backend',lambda a,b:setattr(state,'forced','torch'))
check('torch_compile',lambda a,b:setattr(a._fused_qk_norm,'is_torch_compile',True))
check('missing_norm',lambda a,b:setattr(a,'_fused_qk_norm',None))
check('fa_version',lambda a,b:setattr(state.backend,'fa_impl_ver',4))
check('draft',lambda a,b:setattr(state.backend,'is_draft_runner',True))
check('cascade',lambda a,b:setattr(state.backend,'topk',2))
check('unified_memory',lambda a,b:setattr(state.memory,'enable_unified_memory',True))
check('attn_dp',lambda a,b:setattr(state.parallel,'attn_dp_enabled',True))
check('context_parallel',lambda a,b:setattr(state,'cp',True))
check('dcp_mask',lambda a,b:setattr(b,'dcp_kv_mask','mask'))
check('nonfp8',lambda a,b:setattr(state.backend,'kv_cache_dtype','bf16'))
check('key_scale',lambda a,b:setattr(a.attn,'k_scale',1.))
check('value_scale',lambda a,b:setattr(a.attn,'v_scale',2.))
check('missing_pool',lambda a,b:setattr(state.backend,'buffers',None))
check('pool_stride',lambda a,b:setattr(state.backend.buffers[0],'contiguous',False))
check('cold_capture',lambda a,b:setattr(state,'capture',True))
a,b=reset();assert hook(a,'qkv','pos',b)=='fused_q';state.capture=True
assert hook(a,'qkv','pos',b)=='fused_q';records.append(dict(case='warm_capture_dispatch',result='fused_q'))
# Execute the actual source branch with sentinel tensors to prove no double write.
source=here/'integration/sources/python/sglang/srt/models/muse_spark.py'
modeltree=ast.parse(source.read_text())
branch=next(n for n in ast.walk(modeltree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='fused_q')
calls=[];self=S(attn=lambda *a,**kw:calls.append((a,kw)) or 'attention_output')
namespace=dict(self=self,fused_q='fp8q',forward_batch='batch')
exec(compile(ast.Module(body=[branch],type_ignores=[]),'actual_model_branch','exec'),namespace)
assert calls==[(('fp8q',None,None,'batch'),dict(save_kv_cache=False))]
records.append(dict(case='model_fused_branch',calls=calls))
doc=dict(scope=__doc__,passed=True,records=records)
(here/'host_guard_cpu.json').write_text(json.dumps(doc,indent=2));print('PASS',len(records),'CPU structural controls')
