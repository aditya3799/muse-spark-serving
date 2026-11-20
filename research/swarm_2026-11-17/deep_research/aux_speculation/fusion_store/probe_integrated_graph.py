"""Actual fused kernel inside extracted decoder/model and actual BCG/dedup."""
import argparse,ast,contextlib,copy,importlib.util,json,sys,types,time
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
from runtime_imports import prepare,S
prepare();P=Path(__file__).resolve().parent;A=P.parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def method(path,cls,name,env=None):
    tree=ast.parse(path.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=copy.deepcopy(next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==name));fn.decorator_list=[]
    mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(mod)
    ns={'torch':torch,**(env or {})};exec(compile(mod,str(path),'exec'),ns);return ns[name]
checks=[];report={'started':time.time(),'scope':'Actual JIT arithmetic in extracted decoder/model, real BCG and dedup; fixture attention/MLP/final norm, no full serving runtime.','checks':checks}
def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
def check(name,ok):
    checks.append({'name':name,'passed':bool(ok)});save();assert ok,name
def same(x,y):return torch.equal(x.view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32),y.view(torch.int16 if y.dtype==torch.bfloat16 else torch.int32))
basejit=load('sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit',S/'kernels/ops/layernorm/muse_spark_norm_residual_jit.py')
auxjit=load('sglang.kernels.ops.layernorm.muse_spark_norm_residual_aux_jit',P/'norm_residual_aux.py')
utils=sys.modules['sglang.srt.utils'];utils.get_device_module=lambda:torch.cuda;utils.is_hip=lambda:False;utils.is_xpu=lambda:False;utils.get_bool_env_var=lambda name:False
runtime=types.ModuleType('sglang.srt.runtime_context');runtime.get_exec=lambda:None;sys.modules[runtime.__name__]=runtime
graphdir=S/'srt/model_executor/runner_backend_utils/breakable_cuda_graph'
load('sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils',graphdir/'cuda_utils.py')
bcg=load('fusion_actual_bcg',graphdir/'breakable_cuda_graph.py')
dedup=load('fusion_actual_dedup',S/'srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py')
newroot=P/'integration_sources/python/sglang';oldroot=A/'base_sources/python/sglang'
helper=load('integrated_fused_owner',newroot/'srt/layers/prefill_aux_output.py')
opjit=method(newroot/'kernels/ops/layernorm/muse_spark_norm_residual.py','FusedPostNormResidualPreNorm','forward_jit')
torch.manual_seed(882);hidden=8192;maxrows=256;ids=[0,2,4,6,8]
weights=[torch.rand(hidden,device='cuda').to(torch.bfloat16)+.5 for _ in range(2)]
alpha=torch.full((hidden,),.75,device='cuda');beta=torch.full((hidden,),1.25,device='cuda')
class Boundary:
    _triton_supported=lambda *args:True
    jit_kernel_kwargs=lambda *args:dict(post_eps=1e-6,pre_eps=1e-6,post_weight=weights[0],pre_weight=weights[1],alpha=alpha,beta=beta)
    def __call__(self,*args,**kwargs):return opjit(self,*args,**kwargs)
env={'_use_split_bf16_router':lambda rows:False}
olddecoder=method(oldroot/'srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',env)
newdecoder=method(newroot/'srt/models/muse_spark.py','MuseSpark1xDecoderLayer','forward',env)
class Layer:
    def __init__(self,i,modified):
        self.i=i;self._router_dtype=None;self.operator_dtype=torch.bfloat16;self.is_moe_layer=False
        self._fused_attn_boundary=Boundary();self._fused_ffn_boundary=Boundary() if i<8 else None
        self.self_attn_input_norm=lambda x:x.to(torch.bfloat16)
        self.mlp=lambda x:(x*.75).to(torch.bfloat16)
        self.post_feedforward_layernorm=lambda x:torch.nn.functional.rms_norm(x,(hidden,),eps=1e-6)
        self.post_feedforward_residual_gate=lambda x,y:alpha*x+beta*y.float()
        self.forward=MethodType(newdecoder if modified else olddecoder,self)
    def self_attn(self,pos,x,fb):
        if self.i%3==0:bcg.break_graph()
        return (x*.5+.125).to(torch.bfloat16)
    def __call__(self,*args,**kwargs):return self.forward(*args,**kwargs)
inputs=torch.empty(maxrows,hidden,dtype=torch.bfloat16,device='cuda');inputs.fill_(.25)
variants={}
for kind in ['baseline','candidate']:
    modified=kind=='candidate';root=newroot if modified else oldroot
    forward=method(root/'srt/models/muse_spark.py','MuseSpark1xModel','forward',{'direct_aux_capture':helper.direct_aux_capture,'layernorm_sp':SimpleNamespace(runs_sp=lambda mode:False),'get_forward':lambda:SimpleNamespace(set=lambda *args:None)})
    model=SimpleNamespace(embed_norm=None,layers=[Layer(i,modified) for i in range(9)],layers_to_capture=ids,residual_dtype=torch.float32,operator_dtype=torch.bfloat16,norm=lambda x:x)
    owner=helper.PrefillAuxOutputOwner(maxrows,hidden,ids,'cuda') if modified else None
    class Backend:pass
    for name in ['_alloc_full_buffer','_copy_output_to_buffer','_slice_output']:
        setattr(Backend,name,method(root/'srt/model_executor/runner_backend/breakable_cuda_graph_backend.py','BreakableCudaGraphBackend',name,{'PPProxyTensors':type('PPProxyTensors',(),{})}))
    backend=Backend();backend._cuda_graph_runner=SimpleNamespace(adopt_owned_cuda_graph_output=owner.adopt if owner else lambda *args:None)
    capturestream=torch.cuda.Stream();capturestream.wait_stream(torch.cuda.current_stream());pool=torch.cuda.graph_pool_handle();registry=dedup.DedupedCudaGraphRegistry();graphs={};outputs={};shared=None
    if owner:capgen=owner.begin_capture()
    for rows in [256,17,8]:
        fb=SimpleNamespace(aux_hidden_states_buffer=owner.view(rows) if owner else None,forward_mode=SimpleNamespace(is_extend=lambda:True))
        with torch.cuda.stream(capturestream):
            warm=forward(model,None,None,fb,inputs[:rows])
            if shared is None:shared=backend._alloc_full_buffer(warm,maxrows)
            graph=bcg.BreakableCUDAGraph(registry)
            with bcg.BreakableCUDAGraphCapture(graph,pool=pool,stream=capturestream):
                ret=forward(model,None,None,fb,inputs[:rows]);backend._copy_output_to_buffer(ret,shared,rows)
        graphs[rows]=graph;outputs[rows]=backend._slice_output(shared,rows)
    registry.seal()
    if owner:owner.end_capture(capgen,capturestream)
    torch.cuda.current_stream().wait_stream(capturestream)
    variants[kind]=dict(owner=owner,graphs=graphs,outputs=outputs,registry=registry,stats=registry.stats())
report['dedup']={k:v['stats'] for k,v in variants.items()}
row=torch.arange(maxrows,device='cuda').view(-1,1).remainder(11)/8
col=torch.arange(hidden,device='cuda').view(1,-1).remainder(17)/16
for cycle,rows in enumerate([8,256,17,8,17,256]):
    inputs[:rows].copy_(row[:rows]+col+cycle*.125);snapshots={}
    for kind,v in variants.items():
        owner=v['owner']
        if owner:token=owner.begin_produce()
        v['graphs'][rows].replay();hiddenout,aux=v['outputs'][rows]
        packed=torch.cat(aux,-1) if isinstance(aux,list) else aux
        if owner:owner.end_produce(token,packed)
        with owner.consume(packed) if owner else contextlib.nullcontext():snapshots[kind]=(hiddenout.clone(),packed.clone())
    check('cycle%d_hidden'%cycle,same(snapshots['baseline'][0],snapshots['candidate'][0]))
    for tap in range(5):check('cycle%d_tap%d'%(cycle,tap),same(snapshots['baseline'][1][:,tap*hidden:(tap+1)*hidden],snapshots['candidate'][1][:,tap*hidden:(tap+1)*hidden]))
for v in variants.values():v['registry'].close()
report.update(status='PASS',check_count=len(checks),finished=time.time());save();print(json.dumps({'status':'PASS','checks':len(checks)}))
