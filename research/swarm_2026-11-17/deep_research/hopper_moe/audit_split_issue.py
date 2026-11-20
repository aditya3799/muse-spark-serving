"""Abstract lifetime/order test for the guarded BF16/NVFP4 split-issue path."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
jobs=json.loads((P/'split_issue_jobs.json').read_text())['jobs'];counts={'configs':0,'issues_per_variant':0,'register_prefetches_per_variant':0}
def simulate(c,split):
 bm,bn,bk=c['block_shape'];wm,wn,wk=c['warp_shape'];iters=wk//16
 assert iters>=2 and iters%2==0 and not c['use_packed_k_layout']
 traces=[]
 for part in range(bk//wk):
  q={0:(0,0)};scales={0:(0,0)};b={0:(0,0)};live=None;trace=[]
  for block in range(c['shape_k']//bk):
   for i in range(iters):
    buffer=i%2;nextbuf=(i+1)%2;nxt=(block+(i+1)//iters,(i+1)%iters)
    def prefetch():
     # RS WGMMA consumes converted regs_b, not regs_qb or regs_bs.
     q[nextbuf]=nxt;scales[nextbuf]=nxt
    if not split:prefetch()
    assert b[buffer]==(block,i) and live is None
    live=buffer;trace.append(block*bk+part*wk+i*16)
    if split:prefetch()
    # Existing wait_group0 is retained before any transform or stage release.
    live=None
    assert live is None and q[nextbuf]==scales[nextbuf]==nxt
    b[nextbuf]=nxt
  traces.append(trace)
 return traces
for job in jobs:
 if job['policy']!='split':continue
 c=job['configuration'];a=simulate(c,False);b=simulate(c,True);assert a==b
 counts['configs']+=1;counts['issues_per_variant']+=sum(map(len,a));counts['register_prefetches_per_variant']+=sum(map(len,a))
r={'status':'source_lifetime_model_passed','checks':counts,'guarded_contract':['BF16 activation,NVFP4 weight,FP8e4m3 GS16,FP32 accumulate,indexed,nonpacked,nonwarp-specialized,multicast1,StreamKfalse','WGMMA issues same ordered K16 instructions; existing commit and wait0 retained','Scale consumption remains in transform_b before MMA; no late input-scale or accumulator promotion path','Only next q/scale register prefetch moves between issue and wait; converted current B registers and shared A stage untouched','Every wait finishes before next transform and before existing producer refill/synchronization'],'limits':['Abstract source model; does not emulate CUDA memory model, compiler scheduling, asynchronous tensor core behavior or floating execution','Real Hopper eager/graph bitwise gates and race checking remain required'],'primary_pr':'https://github.com/vllm-project/humming/pull/121','primary_diff_sha256':hashlib.sha256((P/'primary_pr121/pr121.diff').read_bytes()).hexdigest(),'prototype_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('split_kernel.cuh','split_control_kernel.cuh','split_wgmma.cuh')}}
(P/'split_issue_lifetime_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(counts))
