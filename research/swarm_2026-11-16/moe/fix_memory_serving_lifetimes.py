from pathlib import Path
import ast
p=Path(__file__).resolve().parent/'memory_probe.py';s=p.read_text(encoding='utf-8')
s=s.replace("p.add_argument('--tokens',nargs='+',type=int,default=[1,8,32,128,256]);", "p.add_argument('--tokens',nargs='+',type=int,default=[1,8,32,128,256]);p.add_argument('--graph-mode',choices=['serving','independent'],default='serving');")
s=s.replace("original_refs=[weakref.ref(t)", "# Mirror loader staging: retain original tensor storages through the entire\n# module preparation in BOTH modes, then release at simulated context exit.\nloader_originals=[t.detach() for t in raw.parameters()]\noriginal_refs=[weakref.ref(t)")
s=s.replace("prepare_reserved_peak=torch.cuda.max_memory_reserved();gc.collect()", "prepare_reserved_peak=torch.cuda.max_memory_reserved();del loader_originals;gc.collect()")
s=s.replace("'measurement_scope':'PyTorch allocator allocated/reserved only; external CUDA driver/module graph memory not measured',", "'measurement_scope':'One local MoE layer only; PyTorch allocator metrics exclude external CUDA driver/module allocations and full-model graph lifetime effects','loader_lifetime':'Original stock and Humming storages retained through entire preparation, released at simulated loader context exit','graph_mode':a.graph_mode,")
old="graphs=[]\nfor m in a.tokens:\n x=torch.randn((m,h),device='cuda',dtype=torch.bfloat16)/20\n ids=torch.randint(0,e*8,(m,8),device='cuda',dtype=torch.int32)\n scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)"
new="""graphs=[]
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
  scores=torch.rand((m,8),device='cuda');scores/=scores.sum(-1,keepdim=True)"""
assert old in s;s=s.replace(old,new)
s=s.replace("graph,output=capture(torch,call)", "graph,output=capture_current(call)")
s=s.replace("'m':m,'before_allocated'", "'m':m,'graph_pool':list(graph.pool()),'before_allocated'")
s=s.replace("len(report['graphs'])==len(a.tokens)", "len(report['graphs'])==len(capture_order)")
ast.parse(s);p.write_bytes(s.encode('utf-8'));print('serving-like memory mode prepared; no execution')
