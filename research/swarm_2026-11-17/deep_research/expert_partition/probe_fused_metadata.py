from pathlib import Path
import argparse,collections,hashlib,importlib.util,json,random,statistics,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'swarm_2026-11-16'
sys.path.insert(0,str(OLD/'moe'));sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--sglang-root',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
from kernel_runtime import bootstrap
bootstrap(a.sglang_root)
import torch
original_path=OLD/'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
name='sglang.kernels.ops.moe.muse_spark_humming_split';spec=importlib.util.spec_from_file_location(name,original_path);original=importlib.util.module_from_spec(spec);sys.modules[name]=original;spec.loader.exec_module(original)
import scalable_partition as old
import fused_partition as new
report={'status':'running','scope':'Exact metadata partition and CUDA graph tests, plus component latency only','candidate_sha256':hashlib.sha256((P/'fused_partition.py').read_bytes()).hexdigest(),'checks':[],'timings':[]}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.out/'results.json')
def values(n,e,kind):
 rng=random.Random(n+e)
 if kind=='remote':return [-1]*n
 if kind=='hot':return [e-1]*n
 if kind=='invalid':return [(-2,-1,0,e-1,e,2**35)[i%6] for i in range(n)]
 return [rng.randrange(-1,e) for _ in range(n)]
def check(ids,vs,e,out,context):
 cnt=collections.Counter(v for v in vs if 0<=v<e)
 gold=([v if 0<=v<e and cnt[v]<=40 else -1 for v in vs],[v if 0<=v<e and cnt[v]>40 else -1 for v in vs],[cnt[i] for i in range(e)])
 flags=[torch.equal(v,torch.tensor(g,device='cuda',dtype=torch.int32).reshape(v.shape)) for v,g in zip(out,gold)]
 row={**context,'passed':all(flags)};report['checks'].append(row);save();assert row['passed'],row
save()
for e in (1,2,32,256):
 for n in (4096,4097,6024,8192,8193,16384,16385,131072):
  for dtype in (torch.int32,torch.int64):
   for kind in ('remote','hot','invalid','random'):
    vs=values(n,e,kind)
    if dtype==torch.int32:vs=[-3 if v==2**35 else v for v in vs]
    ids=torch.tensor(vs,device='cuda',dtype=dtype);out=new.partition_routes_scalable(ids,e,40)
    check(ids,vs,e,out,{'experts':e,'routes':n,'dtype':str(dtype),'kind':kind,'mode':'eager'})
   if n in (6024,16384):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):go=new.partition_routes_scalable(ids,e,40)
    for kind in ('hot','remote','random'):
     vs=values(n,e,kind);ids.copy_(torch.tensor(vs,device='cuda',dtype=dtype));g.replay();check(ids,vs,e,go,{'experts':e,'routes':n,'dtype':str(dtype),'kind':kind,'mode':'changed_graph'})
    del g,go
   del ids,out
  print(json.dumps({'stage':'metadata_done','e':e,'n':n}),flush=True)
for e in (1,2,32,256):
 for n in (4097,8192,16384):
  ids=torch.tensor(values(n,e,'random'),device='cuda',dtype=torch.int32)
  graphs=[]
  for fn in (old.partition_routes_scalable,new.partition_routes_scalable):
   for _ in range(3):fn(ids,e,40)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=fn(ids,e,40)
   graphs.append((g,out))
  for _ in range(20):
   for g,_ in graphs:g.replay()
  samples={x:[] for x in ('A1','B1','A2','B2')}
  for phase in (1,2):
   for i in range(40):
    for key,index in ((('A',0),('B',1)) if i%2==0 else (('B',1),('A',0))):
     start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graphs[index][0].replay();end.record();end.synchronize();samples[key+str(phase)].append(start.elapsed_time(end)*1000)
  med={k:statistics.median(v) for k,v in samples.items()};controls={x:med[x+'1']/med[x+'2'] for x in ('A','B')}
  report['timings'].append({'e':e,'n':n,'samples_us':samples,'medians_us':med,'controls':controls,'noise_pass':all(.9<=v<=1.1 for v in controls.values()),'speedup':statistics.median(samples['A1']+samples['A2'])/statistics.median(samples['B1']+samples['B2'])});save()
  del graphs,ids,out,g
report['status']='completed';save();print(json.dumps({'status':'completed','checks':len(report['checks'])}),flush=True)

