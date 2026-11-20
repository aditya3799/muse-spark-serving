"""Exact extracted model/decoder/backend wiring; CPU fixture arithmetic."""
import ast,copy,importlib.util,json,sys,types
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
P=Path(__file__).resolve().parent;A=P.parent;checks=[];calls=[]
def check(name,ok):assert ok,name;checks.append({'name':name,'passed':bool(ok)})
def method(path,cls,name,env=None):
    tree=ast.parse(path.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=copy.deepcopy(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name));fn.decorator_list=[]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(mod)
    ns={'torch':torch,**(env or {})};exec(compile(mod,str(path),'exec'),ns);return ns[name]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
newroot=P/'integration_sources/python/sglang';oldroot=A/'candidate_sources/python/sglang'
helper=load('fusion_owner',newroot/'srt/layers/prefill_aux_output.py')
basehelper=load('copy_owner',oldroot/'srt/layers/prefill_aux_output.py')
op_path=newroot/'kernels/ops/layernorm/muse_spark_norm_residual.py'
native=method(op_path,'FusedPostNormResidualPreNorm','forward_native');jit=method(op_path,'FusedPostNormResidualPreNorm','forward_jit')
class TritonKernel:
    def __getitem__(self,grid):
        def run(branch,stream,output,new_stream,router,*args,**kwargs):
            calls.append('triton');new_stream.copy_(stream+branch.float());output.copy_(new_stream*.5)
        return run
triton_fn=method(op_path,'FusedPostNormResidualPreNorm','forward_triton',{'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),'_fused_norm_residual_config':lambda *args:(4,1,{}),'_fused_norm_residual_triton_kernel':TritonKernel()})

def fake_jit(branch,stream,aux_out=None,aux_slot=0,**kwargs):
    calls.append('aux' if aux_out is not None else 'original')
    new=stream+branch.float();out=(new*.5).to(torch.bfloat16)
    if aux_out is not None:aux_out[:,aux_slot*stream.shape[1]:(aux_slot+1)*stream.shape[1]].copy_(new)
    return out,new,None
for module in ['muse_spark_norm_residual_jit','muse_spark_norm_residual_aux_jit']:
    name='sglang.kernels.ops.layernorm.'+module;m=types.ModuleType(name);m.fused_norm_residual=fake_jit;sys.modules[name]=m
class Boundary:
    def __init__(self,kind):
        self.kind=kind;self._post_norm=lambda x:x;self._residual_add=lambda x,y:x+y.float();self._pre_norm=lambda x:x*.5
        self._triton_supported=lambda *args:self.kind!='fallback';self.jit_kernel_kwargs=lambda *args:{}
        self._hidden_size=64;self._use_residual_gate=False;self._post_norm_is_input_scale=False
        self._post_norm.eps=self._pre_norm.eps=1e-6;self._post_norm.has_weight=self._pre_norm.has_weight=False
        for name in ['post_weight','pre_weight','alpha','beta','empty_fp32']:setattr(self,'_cc_'+name,torch.empty(0))
        self._ensure_cached_weights=lambda *args:None
    def forward_native(self,*args,**kwargs):return native(self,*args,**kwargs)
    def __call__(self,*args,**kwargs):return (native if self.kind=='native' else triton_fn if self.kind=='triton' else jit)(self,*args,**kwargs)
def gather(comm,branch,stream,**kwargs):
    calls.append('gather');new=stream+branch.repeat(1,2).float();return (new*.5).to(torch.bfloat16),new
env={'_use_split_bf16_router':lambda n:False,'all_gather_norm_residual':gather,'get_parallel':lambda:SimpleNamespace(tp_group=SimpleNamespace(ca_comm=None))}
new_decoder=method(newroot/'srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',env)
old_decoder=method(oldroot/'srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',env)
class Layer:
    def __init__(self,i,kind,modified):
        self.i=i;self.self_attn_input_norm=lambda x:x
        self.self_attn=lambda pos,x,fb:(x*.125).to(torch.bfloat16)
        self._router_dtype=None;self._fused_attn_boundary=Boundary(kind);self.is_moe_layer=False
        self.mlp=lambda x:((x[:,:32] if i==4 else x)*.25).to(torch.bfloat16)
        self._fused_ffn_boundary=None if i==8 else Boundary(kind)
        self.operator_dtype=torch.bfloat16;self.post_feedforward_layernorm=lambda x:x
        self.post_feedforward_residual_gate=lambda x,y:x+y.float()
        self.forward=MethodType(new_decoder if modified else old_decoder,self)
    def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)
def model_fn(root,h):
    return method(root/'srt/models/muse_spark.py','MuseSpark1xModel','forward',{'direct_aux_capture':h.direct_aux_capture,'get_forward':lambda:SimpleNamespace(set=lambda *args:None),'layernorm_sp':SimpleNamespace(runs_sp=lambda mode:False)})
old_forward=model_fn(oldroot,basehelper);new_forward=model_fn(newroot,helper)
for kind in ['native','jit','fallback','triton']:
    for rows in [1,7,33]:
        torch.manual_seed(rows);x=torch.randn(rows,64).to(torch.bfloat16)
        def model(modified):return SimpleNamespace(embed_norm=None,layers=[Layer(i,kind,modified) for i in range(9)],layers_to_capture=[0,2,4,6,8],residual_dtype=torch.float32,operator_dtype=torch.bfloat16,norm=lambda x:x)
        for extend in [False,True]:
            out=torch.full((rows,64*5),-77,dtype=torch.bfloat16)
            fb=SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda:extend),aux_hidden_states_buffer=out)
            ref=old_forward(model(False),None,None,fb,x)
            got=new_forward(model(True),None,None,fb,x)
            name=kind+'_m%d_extend%s'%(rows,extend)
            check(name+'_hidden',torch.equal(ref[0],got[0]))
            ra=torch.cat(ref[1],-1) if isinstance(ref[1],list) else ref[1]
            ga=torch.cat(got[1],-1) if isinstance(got[1],list) else got[1]
            check(name+'_aux',torch.equal(ra,ga))
            if extend:check(name+'_owner',got[1].data_ptr()==out.data_ptr())
check('triton_path_exercised','triton' in calls)
check('original_jit_path_exercised','original' in calls);check('aux_jit_path_exercised','aux' in calls);check('gather_fallback_exercised','gather' in calls)
(P/'cpu_integration_results.json').write_text(json.dumps({'status':'PASS','checks':checks,'check_count':len(checks),'scope':'Exact extracted model, decoder, native/JIT backend methods. CPU fixture attention/MLP and JIT kernels; no true GPU arithmetic or full model.'},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
