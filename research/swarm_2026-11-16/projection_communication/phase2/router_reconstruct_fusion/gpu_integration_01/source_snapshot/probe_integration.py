"""Exact model router caller with real BaseFusedOp dispatch and candidate CUDA."""
from pathlib import Path
import argparse,ast,contextlib,hashlib,importlib.util,json,os,sys,types
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ.get('RECONSTRUCT_BOUND')=='240';a.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'integration_manifest.json').read_text())
for rel,v in manifest['sources'].items():assert sha(P/'integration_sources'/rel)==v['candidate']
sys.path.insert(0,str(a.workspace/'optimization_marlin_tiles_2026-11-15'))
from kernel_runtime import bootstrap
provenance=bootstrap(a.workspace/'sglang')
import torch
from sglang.kernels.fused_op import BaseFusedOp,set_fused_op_backend
from sglang.kernels.spec import CapabilityRequirement,KernelBackend
from sglang.kernels.ops.gemm.muse_spark_router_gemm import MAX_TOKENS
from sglang.kernels.ops.moe.muse_spark_routing_jit import fused_muse_spark_routing_jit as stock_jit
q=P/'integration_sources/python/sglang/kernels/ops/moe/muse_spark_routing_jit.py'
spec=importlib.util.spec_from_file_location('integration_router_jit',q);jit=importlib.util.module_from_spec(spec);spec.loader.exec_module(jit)
loader=jit.load_jit
header=P/'integration_sources/python/sglang/kernels/jit/csrc/muse_spark/routing.cuh'
def local_load(*args,**kw):
    assert kw['cuda_files']==['muse_spark/routing.cuh'];kw['cuda_files']=[str(header)];return loader(*args,**kw)
jit.load_jit=local_load
sum_calls=[];actual_jit=jit.fused_muse_spark_routing_jit
def observed_jit(*args,**kw):
    sum_calls.append(kw.get('sum_halves',False));return actual_jit(*args,**kw)
jit.fused_muse_spark_routing_jit=observed_jit
sys.modules['sglang.kernels.ops.moe.muse_spark_routing_jit']=jit
# Exact class/function source nodes skip unrelated serving module initializers.
# Native fallback uses the default custom-router-sum=false configuration.
ns={'__name__':'isolated_integrated_routing','torch':torch,'BaseFusedOp':BaseFusedOp,'CapabilityRequirement':CapabilityRequirement,'KernelBackend':KernelBackend,'envs':types.SimpleNamespace(SGLANG_USE_CUSTOM_ROUTER_SUM=types.SimpleNamespace(get=lambda:False))}
common=a.workspace/'sglang/python/sglang/srt/utils/common.py';topk=next(n for n in ast.parse(common.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fast_topk')
exec(compile(ast.Module(body=[topk],type_ignores=[]),str(common),'exec'),ns)
route=P/'integration_sources/python/sglang/kernels/ops/moe/muse_spark_routing.py'
rt=ast.parse(route.read_text());nodes=[n for n in rt.body if (isinstance(n,ast.ClassDef) and n.name=='FusedMuseSparkRouting') or (isinstance(n,ast.FunctionDef) and n.name=='pack_topk_ids_native')]
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[]),str(route),'exec'),ns);Op=ns['FusedMuseSparkRouting']
orig_ns=dict(ns);original_route=a.workspace/'sglang/python/sglang/kernels/ops/moe/muse_spark_routing.py';orig_cls=next(n for n in ast.parse(original_route.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FusedMuseSparkRouting')
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[orig_cls],type_ignores=[]),str(original_route),'exec'),orig_ns);OriginalOp=orig_ns['FusedMuseSparkRouting']
model=P/'integration_sources/python/sglang/srt/models/muse_spark.py';mt=ast.parse(model.read_text());router=next(n for n in mt.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xRouter');moe=next(n for n in mt.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xMoE')
methods=[n for n in moe.body if isinstance(n,ast.FunctionDef) and n.name in ('_routing_op','_forward_experts')]
state=types.SimpleNamespace(sp_active=False)
mn={'__name__':'isolated_integrated_model','torch':torch,'nn':torch.nn,'F':torch.nn.functional,'contextlib':contextlib,'_is_cpu':False,'_split_bf16_router_supported':lambda:True,'ROUTER_GEMM_MAX_TOKENS':MAX_TOKENS,'get_forward':lambda:state,'get_global_experts_capturer':lambda:None,'FusedMuseSparkRouting':Op}
policy=next(n for n in mt.body if isinstance(n,ast.FunctionDef) and n.name=='_use_split_bf16_router')
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[policy,router]+methods,type_ignores=[]),str(model),'exec'),mn);Router=mn['MuseSpark1xRouter']
class ExpertOutputSpy:
    use_marlin=True;post_expert_norm=None
    def __call__(self,x,scores,ids,*args,**kwargs):return scores,ids,kwargs['packed_topk_ids']
torch.set_grad_enabled(False);torch.manual_seed(1901)
gate=Router(8192,256,torch.float32).cuda();gate.weight.normal_(0,1/90)
side=torch.cuda.Stream();owner=types.SimpleNamespace(gate=gate,router_in_fp32=True,nvfp4_checkpoint=True,pre_expert_norm=None,pre_expert_proj=None,experts=ExpertOutputSpy(),expert_hidden_size=8192,top_k=8,route_scale=1.,router_multiplier=.17677669529663687,route_eps=1e-15,layer_idx=0,_side_stream=lambda:side)
for name in ('_routing_op','_forward_experts'):setattr(owner,name,types.MethodType(mn[name],owner))
refop=OriginalOp(gate.e_score_correction_bias,1.,.17677669529663687,1e-15)
report={'status':'running','scope':'Exact extracted model _forward_experts/_routing_op and Router source, real BaseFusedOp dispatch and candidate JIT/CUDA; pre-projection disabled and experts replaced by output sink. Native fallback config custom-router-sum=false. Hopper architecture support predicate forced true on local GPU to exercise unchanged token policy. Not full SGLang server, MoE, H200 or collectives.','architecture_gate_simulated':True,'actual_capability':torch.cuda.get_device_capability(),'original_token_threshold':MAX_TOKENS,'sources':manifest['sources'],'gpu':torch.cuda.get_device_name(),'checks':[],'harness_sha256':sha(Path(__file__))}
def save():
    t=a.out/'report.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.out/'report.json')
def reference(x,backend):
    logits=gate.forward_split2(x)
    if backend==KernelBackend.TORCH:return refop.forward_native(None,logits,8,True,want_packed=True)
    return stock_jit(logits,gate.e_score_correction_bias,8,1.,router_output_multiplier=.17677669529663687,router_score_division_eps=1e-15,packed_topk_ids=torch.empty((x.shape[0],8),device='cuda',dtype=torch.int32))
def check(label,result,expected,**ctx):
    mismatches=[int(((x.view(torch.int32) if x.dtype==torch.float32 else x)!=(y.view(torch.int32) if y.dtype==torch.float32 else y)).sum().item()) for x,y in zip(result,expected)]
    row={'label':label,'context':ctx,'mismatches_weights_ids_packed':mismatches,'passed':not any(mismatches)};report['checks'].append(row);save()
    if not row['passed']:report.update(status='failed',failure=row);save();raise AssertionError(row)
save()
for backend in (None,KernelBackend.JIT,KernelBackend.TORCH):
    set_fused_op_backend(backend)
    for m in (64,512,4096):
        for dtype in (torch.bfloat16,torch.float16):
            x=torch.randn((m,8192),device='cuda',dtype=dtype)*.05
            for pattern in ('random','near_ties'):
                gate.weight.normal_(0,1/90)
                if pattern=='near_ties':gate.weight[1::2].copy_(gate.weight[::2]);gate.weight[1::2,0].add_(1e-7)
                sum_calls.clear();result=owner._forward_experts(x);expected=reference(x,backend);check('eager',result,expected,backend=str(backend),m=m,dtype=str(dtype),pattern=pattern)
                if backend!=KernelBackend.TORCH:assert sum_calls and all(sum_calls)
                else:assert not sum_calls
            for _ in range(3):owner._forward_experts(x)
            torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):actual=owner._forward_experts(x)
            for scale in (.05,.25,0.):
                x.copy_(torch.randn_like(x)*scale);g.replay();check('graph_changed',actual,reference(x,backend),backend=str(backend),m=m,dtype=str(dtype),scale=scale)
            del g
        print(json.dumps({'milestone':'shape_complete','backend':str(backend),'m':m}),flush=True)
set_fused_op_backend(None);report.update(status='completed',passed=True,actual_source_model_path=True);save();print(json.dumps({'status':'completed','checks':len(report['checks'])}),flush=True)
