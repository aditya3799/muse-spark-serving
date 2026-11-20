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
import oneshot_partition as new
import triton
report={'status':'running','scope':'Partition numerical, changed-graph and canary checks under separately recorded Compute Sanitizer tool. Racecheck covers shared hazards, not global inter-CTA ownership proof.','rows':[]}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.out/'results.json')
def compare(out,values,e,context):
 cnt=collections.Counter(v for v in values if 0<=v<e)
 gold=([v if 0<=v<e and cnt[v]<=40 else -1 for v in values],[v if 0<=v<e and cnt[v]>40 else -1 for v in values],[cnt[i] for i in range(e)])
 flags=[torch.equal(v,torch.tensor(g,device='cuda',dtype=torch.int32).reshape(v.shape)) for v,g in zip(out,gold)]
 row={**context,'reference_equal':all(flags)};report['rows'].append(row);save();assert all(flags),row
save()
for e,n in ((1,4097),(32,8193),(32,16384),(256,4097),(32,16385),(32,131072)):
 backing=torch.empty((n+2,),device='cuda',dtype=torch.int64);ids=backing[1:-1]
 values=[(-2,-1,0,e-1,e,2**35)[i%6] for i in range(n)];ids.copy_(torch.tensor(values,device='cuda',dtype=torch.int64))
 for label,fn in (('hierarchical',old.partition_routes_scalable),('oneshot',new.partition_routes_scalable)):
  fn(ids,e,40);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fn(ids,e,40)
  for mode in ('invalid','hot','remote'):
   vals=values if mode=='invalid' else [e-1]*n if mode=='hot' else [-1]*n
   ids.copy_(torch.tensor(vals,device='cuda',dtype=torch.int64));g.replay();torch.cuda.synchronize();compare(out,vals,e,{'experts':e,'routes':n,'path':label,'mode':mode})
  del g,out
 if n<=16384:
  canary=-123456789
  lb=torch.full((n+2,),canary,device='cuda',dtype=torch.int32);hb=lb.clone();cb=torch.full((e+2,),canary,device='cuda',dtype=torch.int32)
  ids.copy_(torch.tensor(values,device='cuda',dtype=torch.int64))
  new._expert_owned_partition[(e,)](ids,lb[1:-1],hb[1:-1],cb[1:-1],ROUTES=n,EXPERTS=e,CUTOFF=40,BLOCK=triton.next_power_of_2(n),num_warps=8 if n>8192 else 4)
  torch.cuda.synchronize();assert all(int(buf[i])==canary for buf in (lb,hb,cb) for i in (0,-1))
  compare((lb[1:-1],hb[1:-1],cb[1:-1]),values,e,{'experts':e,'routes':n,'path':'oneshot_offset_canary','mode':'invalid','canaries_intact':True})
  del lb,hb,cb
 del ids,backing
 print(json.dumps({'stage':'case_complete','e':e,'n':n}),flush=True)
report['status']='completed';save();print(json.dumps({'status':'completed','checks':len(report['rows'])}),flush=True)
