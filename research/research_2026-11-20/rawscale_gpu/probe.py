import ast,ctypes,hashlib,json,pathlib,statistics,sys,time
import numpy as np
import torch
R=pathlib.Path('D:/OpenKernelChallenge/serving');P=R/'research_2026-11-19_round2/moe';O=R/'research_2026-11-20/rawscale_gpu';O.mkdir(exist_ok=True)
sys.path.insert(0,str(P));from forward_oracles import wgs_forward
scope={'__file__':str(P/'prove_pr50800_rawscale_layout.py')}
source=(P/'prove_pr50800_rawscale_layout.py').read_text();exec(source[:source.index('\nresults=[]')],scope)
lib=ctypes.CDLL(str(O/'rawscale.dll'));fn=lib.pack_raw;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
report={'scope':'GPU byte-layout converter only; synthetic codes including all nibble/scale encodings; not a MoE GEMM, H200 result, or serving speedup','torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'cases':[],'complete':False}
def save():(O/'result.json').write_text(json.dumps(report,indent=2))
for seed in [20111961,20111967]:
 for E,N in [(1,4096),(1,8192),(2,4096),(2,8192)]:
  rng=np.random.default_rng(seed+N+E);q=rng.integers(0,16,(E,N,4096),dtype=np.uint8)
  sf=np.arange(E*N*256,dtype=np.uint32).astype(np.uint8).reshape(E,N,256);rng.shuffle(sf,axis=1)
  packed=wgs_forward(q,sf,N==8192);expected=scope['forward'](q,sf,N==8192)
  x=torch.from_numpy(packed.copy()).cuda();guard=torch.full((expected.size+512,),165,dtype=torch.uint8,device='cuda');out=guard[256:-256]
  original=x.clone();item={'seed':seed,'experts':E,'N':N,'bytes':expected.size,'blocks':{}};report['cases'].append(item)
  def run(block):assert fn(x.data_ptr(),out.data_ptr(),E,N,block,torch.cuda.current_stream().cuda_stream)==0
  for block in [128,256,512]:
   run(block);torch.cuda.synchronize();actual=out.cpu().numpy().reshape(expected.shape)
   assert np.array_equal(actual,expected) and bool(torch.equal(x,original))
   assert bool((guard[:256]==165).all()) and bool((guard[-256:]==165).all())
   times=[]
   for rep in range(10):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();run(block);b.record();b.synchronize()
    if rep>=2:times.append(a.elapsed_time(b)*1000)
   item['blocks'][str(block)]={'exact':True,'microseconds':times,'median_us':statistics.median(times)}
  restoredq,restoredsf=scope['inverse'](actual,N,N==8192)
  assert np.array_equal(restoredq,q) and np.array_equal(restoredsf,sf)
  # A changed source must be observed on replay, without modifying graph pointers.
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):run(256)
  torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):run(256)
  q^=15;sf^=255;changed=wgs_forward(q,sf,N==8192);x.copy_(torch.from_numpy(changed).cuda());graph.replay();torch.cuda.synchronize()
  assert np.array_equal(out.cpu().numpy().reshape(expected.shape),scope['forward'](q,sf,N==8192))
  item.update(inverse_exact=True,changed_graph_exact=True,guards_exact=True,source_preserved=True);save();print(seed,E,N,{b:round(v['median_us'],2) for b,v in item['blocks'].items()},flush=True)
  del graph,x,original,guard,out,q,sf,packed,expected,actual,restoredq,restoredsf,changed;torch.cuda.empty_cache()
report['complete']=True;report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
