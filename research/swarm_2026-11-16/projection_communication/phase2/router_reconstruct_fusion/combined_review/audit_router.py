"""CPU execute exact model caller and routing handlers with tensor/runtime spies."""
from pathlib import Path
import ast,contextlib,hashlib,json,sys,types
P=Path(__file__).resolve().parent;src=Path('D:\\OpenKernelChallenge\\serving\\swarm_2026-11-16\\projection_communication\\phase2\\router_reconstruct_fusion\\integration_sources')
model=Path('D:\\OpenKernelChallenge\\serving\\swarm_2026-11-16\\coordination\\research_resumed\\norm_glue_router\\candidate_sources\\python\\sglang\\srt\\models\\muse_spark.py');routing=src/'python/sglang/kernels/ops/moe/muse_spark_routing.py'
S=types.SimpleNamespace;events=[]
class Tensor:
    device='fake'
    def __init__(self,shape=(64,256),dtype='fp32'):self.shape=shape;self.dtype=dtype
    def float(self):return Tensor(self.shape)
    def contiguous(self):return self
    def chunk(self,n,dim=-1):
        events.append('add_input_chunk');half=Tensor((self.shape[0],self.shape[1]//2));half.from_split=True;return half,half
    def __add__(self,other):
        events.append('materialized_add' if getattr(self,'from_split',False) else 'other_add');return Tensor(self.shape)
    def __mul__(self,other):return self
    def to(self,*a):return self
    def long(self):return self
    def gather(self,dim,indices):return Tensor(indices.shape)
    def new_zeros(self,shape):return Tensor(shape)
def selected(x,bias,k,*args,**kw):
    events.append(('handler_input',x.shape,kw.get('sum_halves',False)))
    return Tensor((x.shape[0],k)),Tensor((x.shape[0],k),'i32'),Tensor((x.shape[0],k),'i32')
fakejit=types.ModuleType('sglang.kernels.ops.moe.muse_spark_routing_jit');fakejit.fused_muse_spark_routing_jit=selected;sys.modules[fakejit.__name__]=fakejit
class Base:
    is_torch_compile=False;_oot_forward_registry={}
    def backend_eligible(self,*a,**k):return True
    def __call__(self,*a,**kw):return getattr(self,'forward_'+self.backend)(*a,**kw)
    def _finalize(self,w,ids,x,want):return (w,ids,Tensor(ids.shape,'i32')) if want else (w,ids)
ns={'torch':S(Tensor=Tensor,float32='fp32',bfloat16='bf16',float16='fp16',int32='i32',empty=lambda shape,**k:Tensor(shape,k.get('dtype','fp32')),sigmoid=lambda x:x,ops=S(sgl_kernel=S(topk_sigmoid_cpu=lambda x,y,k,*a:selected(x,None,k)[:2]))),
    'BaseFusedOp':Base,'KernelBackend':S(JIT='jit'),'fast_topk':lambda x,k,dim:(Tensor((x.shape[0],k)),Tensor((x.shape[0],k),'i32')),'fused_muse_spark_routing':selected}
tree=ast.parse(routing.read_text(encoding='utf-8'));orig=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FusedMuseSparkRouting')
names={'supports_split2_accumulators','backend_eligible','forward_jit','forward_triton','forward_native','forward_cpu'}
cl=ast.ClassDef(name=orig.name,bases=[ast.Name(id='BaseFusedOp',ctx=ast.Load())],keywords=[],body=[n for n in orig.body if isinstance(n,ast.FunctionDef) and n.name in names],decorator_list=[])
ast.fix_missing_locations(cl);exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[cl],type_ignores=[]),str(routing),'exec'),ns);Op=ns[orig.name]
def op(backend='jit'):
    o=Op();o.backend=backend;o.expert_bias=Tensor((256,));o.route_scale=1.;o.router_output_multiplier=None;o.router_score_normalization='sigmoid';o.router_score_division_eps=1e-15;o.sol_topk_out=None;o.local_expert_mapping=None;o.num_valid_tokens=None;return o
checks=[]
for backend in ('jit','triton','native','cpu'):
    for fused in (False,True):
        events.clear();o=op(backend);output=o(None,Tensor((64,512 if fused else 256)),8,True,want_packed=True,sum_halves=fused)
        assert ('materialized_add' in events)==(fused and backend!='jit'),(backend,events)
        assert output[0].shape==(64,8)
        if backend in ('jit','triton'):
            record=next(e for e in events if isinstance(e,tuple) and e[0]=='handler_input')
            assert record[1]==(64,512 if fused and backend=='jit' else 256)
            assert record[2]==(fused and backend=='jit')
        checks.append({'scope':'exact_handler','backend':backend,'fused':fused,'passed':True,'events':list(events)})
class MuseSpark1xRouter:
    weight=Tensor((256,8192))
    def forward_split2(self,x,*,return_accumulators=False):events.append(('split2',return_accumulators));return Tensor((x.shape[0],512 if return_accumulators else 256))
    def __call__(self,x):events.append(('normal_router',));return Tensor((x.shape[0],256))
class SubRouter(MuseSpark1xRouter):pass
class SubOp(Op):pass
class MuseSpark1xRMSNorm:
    def __call__(self,x):events.append('row_norm');return x
class Experts:
    use_marlin=True;post_expert_norm=None
    def __call__(self,x,*a,**kw):return x
modeltree=ast.parse(model.read_text(encoding='utf-8'));mc=next(n for n in modeltree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xMoE');node=next(n for n in mc.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_experts')
state=S(sp_active=False)
ms={'torch':ns['torch'],'contextlib':contextlib,'_is_cpu':False,'_use_split_bf16_router':lambda m:m>8,'get_forward':lambda:state,'get_global_experts_capturer':lambda:None,'MuseSpark1xRouter':MuseSpark1xRouter,'FusedMuseSparkRouting':Op}
ms['torch'].cuda=S(is_available=lambda:False,get_device_capability=lambda device:(9,0))
ms.update(MuseSpark1xRMSNorm=MuseSpark1xRMSNorm,get_parallel=lambda:S(tp_group=S(ca_comm=None)),gemm_ag_supported=lambda *a:False,prefill_row_projection=lambda g,x,w,row_norm=None:row_norm(x) if row_norm else x)
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[node],type_ignores=[]),str(model),'exec'),ms)
for scenario in ('eligible','small','sp','compile','gate_subclass','op_subclass','oot','no_split','custom_callable','eligible_norm'):
    for backend in ('jit','triton','native'):
        events.clear();state.sp_active=scenario=='sp';o=op(backend)
        if scenario=='op_subclass':o.__class__=SubOp
        o.is_torch_compile=scenario=='compile';o._oot_forward_registry={'external':{Op:object()}} if scenario=='oot' else {}
        if scenario=='custom_callable':
            saved=o;o=S(__call__=None)
            class Custom:
                def __call__(self,*a,**kw):
                    assert 'sum_halves' not in kw;return saved(*a,**kw)
            o=Custom()
        gate=SubRouter() if scenario=='gate_subclass' else MuseSpark1xRouter()
        x=Tensor((8 if scenario=='small' else 64,8192),'bf16')
        owner=S(router_in_fp32=scenario!='no_split',gate=gate,nvfp4_checkpoint=True,pre_expert_norm=MuseSpark1xRMSNorm() if scenario=='eligible_norm' else None,pre_expert_proj=S(weight=object()) if scenario=='eligible_norm' else None,experts=Experts(),tp_size=8,_can_shard_prefill_row_projection=lambda *a:True,expert_hidden_size=4096,top_k=8,_routing_op=lambda:o,_forward_sp_experts=lambda *a:a[0])
        result=ms['_forward_experts'](owner,x)
        assert result is x
        assert events.count('row_norm')==int(scenario=='eligible_norm')
        expect=scenario in ('eligible','eligible_norm')
        assert ('split2',True) in events if expect else ('split2',True) not in events
        checks.append({'scope':'exact_model_caller','scenario':scenario,'backend':backend,'accumulator_selected':expect,'passed':True})
report={'status':'passed','cases':len(checks),'checks':checks,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,routing)},'scope':'Exact source model caller and handler ASTs, CPU spies for tensor ops and BaseFusedOp selection. No real backend selection, GPU arithmetic or full serving import proof.'}
(P/'CPU_INTEGRATION_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(checks)}))
