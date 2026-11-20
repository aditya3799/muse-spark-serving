"""Fresh-process stock/Humming memory probe; run only with coordinated GPU access."""
from pathlib import Path
import argparse,ast,gc,hashlib,json,sys,time,weakref
from types import SimpleNamespace,MethodType
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path.insert(0,str(P/'humming_vendor'));sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True);p.add_argument('--mode',choices=['stock','humming'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',nargs='+',type=int,default=[1,8,32,128,256]);p.add_argument('--graph-mode',choices=['serving','independent'],default='serving');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
from probe_preflight import preflight
provenance=preflight(a.sglang_root)
if not a.tokens or any(m<=0 for m in a.tokens):raise ValueError('Provide positive token sizes')
from kernel_runtime import bootstrap
runtime=bootstrap(a.sglang_root)
import torch
from marlin_bench import load_method,capture
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
export=P/'serving_humming_v1/python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
tree=ast.parse(export.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs')]
ns={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(export),'exec'),ns)
e=h=i=0;e=32;h=i=4096;torch.manual_seed(509)
raw=torch.nn.Module();raw.quant_config=SimpleNamespace(group_size=16);raw.moe_runner_config=SimpleNamespace(is_gated=True);raw.params_dtype=torch.bfloat16;raw.intermediate_size_per_partition=i;raw.num_local_experts=e;raw.global_num_experts=e*8
for stage,n,k in [('w13',2*i,h),('w2',h,i)]:
 w=torch.randint(0,256,(e,n,k//2),device='cuda',dtype=torch.uint8)
 sf=torch.randint(48,64,(e,n,k//16),device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn)
 gs=torch.linspace(.0078125,.03125,e,device='cuda',dtype=torch.float32)
 for suffix,t in [('weight',w),('weight_scale',sf),('weight_scale_2',gs)]:setattr(raw,stage+'_'+suffix,torch.nn.Parameter(t,requires_grad=False))
del w,sf,gs,t
# Mirror loader staging: retain original tensor storages through the entire
# module preparation in BOTH modes, then release at simulated context exit.
loader_originals=[t.detach() for t in raw.parameters()]
original_refs=[weakref.ref(t) for t in raw.parameters()];original_ptrs={t.untyped_storage().data_ptr() for t in raw.parameters()}
torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
raw_bytes=torch.cuda.memory_allocated();started=time.time()
if a.mode=='humming':
 layer=ns['_prepare_humming_experts'](raw,torch);del raw
else:
 prepare_moe_nvfp4_layer_for_marlin(raw);layer=raw;del raw
torch.cuda.synchronize();prepare_peak=torch.cuda.max_memory_allocated();prepare_reserved_peak=torch.cuda.max_memory_reserved();del loader_originals;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
retained_ptrs={t.untyped_storage().data_ptr() for t in list(layer.parameters())+list(layer.buffers())}
report={'mode':a.mode,'status':'running','argv':sys.argv,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry':{'local_experts':e,'global_experts':e*8,'hidden':h,'intermediate':i,'top_k':8,'tokens':a.tokens},'measurement_scope':'One local MoE layer only; PyTorch allocator metrics exclude external CUDA driver/module allocations and full-model graph lifetime effects','loader_lifetime':'Original stock and Humming storages retained through entire preparation, released at simulated loader context exit','graph_mode':a.graph_mode,'runtime':runtime,'provenance':provenance,'scope':'local single-shard tensor/memory validation only; no full-model bound','export_sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'raw_allocated':raw_bytes,'prepare_peak_allocated':prepare_peak,'prepare_peak_reserved':prepare_reserved_peak,'prepare_extra_over_raw':prepare_peak-raw_bytes,'prepare_seconds':time.time()-started,'persistent_allocated':torch.cuda.memory_allocated(),'persistent_reserved':torch.cuda.memory_reserved(),'original_storage_retained':len(original_ptrs&retained_ptrs),'original_parameter_refs_alive':sum(r() is not None for r in original_refs),'graphs':[]}
(a.out/'memory.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if a.mode=='humming':assert not original_ptrs&retained_ptrs
mapping=torch.full((e*8,),-1,device='cuda',dtype=torch.int32);mapping[:e]=torch.arange(e,device='cuda',dtype=torch.int32)
lookup=torch.cat((mapping.new_full((1,),-1),mapping));norm=torch.linspace(.5,1.5,h,device='cuda',dtype=torch.bfloat16)
if a.mode=='stock':
 layer.register_buffer('_expert_map',mapping);layer.register_buffer('_expert_lookup',lookup)
 method=MethodType(load_method(a.sglang_root,'baseline',torch)[0],layer)
graphs=[]
capture_order=sorted(set(a.tokens),reverse=True) if a.graph_mode=='serving' else a.tokens
report['capture_order']=capture_order
if a.graph_mode=='serving':
 pool=torch.cuda.graph_pool_handle();capture_stream=torch.cuda.Stream()
 largest=max(capture_order)
 input_buffer=torch.randn((largest,h),device='cuda',dtype=torch.bfloat16)/20
 route_buffer=torch.randint(0,e*8,(largest,8),device='cuda',dtype=torch.int32)
 score_buffer=torch.rand((largest,8),device='cuda');score_buffer/=score_buffer.sum(-1,keepdim=True)
 report['shared_pool']=list(pool)
 report['input_buffers']='One largest-bucket allocation, sliced for smaller captures'
else:
 report['input_buffers']='Independent per-bucket allocations; separate graph pools, stress evidence only'
def capture_current(call):
 if a.graph_mode=='independent':return capture(torch,call)
 capture_stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(capture_stream):
  for _ in range(3):call()
 torch.cuda.current_stream().wait_stream(capture_stream)
 torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph,pool=pool,stream=capture_stream):output=call()
 return graph,output
for m in capture_order:
 if a.graph_mode=='serving':
  x=input_buffer[:m];ids=route_buffer[:m];scores=score_buffer[:m]
 else:
  x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20
  ids=torch.randint(0,e*8,(m,8),device='cuda',dtype=torch.int32)
  scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
 def call():
  if a.mode=='humming':values,local=ns['_humming_expert_outputs'](layer,x,lookup[ids+1],torch)
  else:values,local=method(x,scores,ids)
  return fused_post_expert_norm_weighted_sum(values,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=local)
 torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();graph,output=capture_current(call)
 graph.replay();torch.cuda.synchronize();assert torch.isfinite(output).all()
 graphs.append((graph,output,x,ids,scores))
 report['graphs'].append({'m':m,'graph_pool':list(graph.pool()),'before_allocated':before,'after_allocated':torch.cuda.memory_allocated(),'after_reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
 (a.out/'memory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['graphs'][-1]),flush=True)
 # Replay changing live buffers after recording allocation metrics. Arithmetic
 # comparison is performed by frozen_helper_probe, not this memory-only probe.
 x.neg_();ids.fill_(e);scores.fill_(1.0/8)
 graph.replay();torch.cuda.synchronize();assert bool((output==0).all())
 ids.zero_();graph.replay();torch.cuda.synchronize();assert bool(torch.isfinite(output).all())
 # Cooperative handoff: callers may request exit at a completed synchronized case.
 if (a.out/'STOP_AFTER_CASE').exists():
  report['stopped_cooperatively']=True
  break
report['completed']=len(report['graphs'])==len(capture_order);report['status']='completed' if report['completed'] else 'stopped_cooperatively';(a.out/'memory.json').write_text(json.dumps(report,indent=2));print('complete',flush=True)
