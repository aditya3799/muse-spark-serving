import pathlib,resource,json,hashlib,statistics,math,random,time,gc
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
mem=dict((x.split(':')[0],int(x.split()[1])) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:')))
if mem['MemAvailable']<2*1024**2:raise RuntimeError('Host memory preflight below 2 GiB')
import torch
from target_select import native_select,fused_select
HERE=pathlib.Path(__file__).resolve().parent

def gm(v):return math.exp(statistics.mean(math.log(x) for x in v))
def summary(records):
 gain=[statistics.mean(r['native'])/statistics.mean(r['candidate']) for r in records]
 control=[statistics.mean(r['native'])/statistics.mean(r['control']) for r in records]
 rng=random.Random(197743);boots=sorted(gm([gain[rng.randrange(len(gain))] for _ in gain]) for _ in range(2000))
 cs=sorted(gm([control[rng.randrange(len(control))] for _ in control]) for _ in range(2000))
 ok=.97<=gm(control)<=1.03
 return dict(gain=gm(gain),gain_ci95=[boots[50],boots[1949]],control_ratio=gm(control),control_ci95=[cs[50],cs[1949]],controls_valid=ok,component_keep=ok and boots[50]>1.05,median_us={k:statistics.median([statistics.mean(r[k]) for r in records]) for k in records[0]})
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report=dict(scope='Only target post-gather selection. No LM head, native softcap, transport, acceptance or serving claim.',source_sha256=hashlib.sha256((HERE/'target_select.py').read_bytes()).hexdigest(),seed=197743,host_memory_kib=mem,complete=False,cases=[])
 path=HERE/'timing.json'
 def save():path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for n in (8,32,128,256):
  torch.manual_seed(197743+n);pairs=torch.randn((8,n,2),device='cuda');pairs[...,1].copy_(torch.randint(0,2**24,(8,n),device='cuda').float())
  funcs=dict(native=native_select,candidate=fused_select,control=native_select);graphs={};outputs={}
  for name,fn in funcs.items():
   fn(pairs);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
    for _ in range(128):out=fn(pairs)
   graphs[name]=g;outputs[name]=out
  for _ in range(20):
   for g in graphs.values():g.replay()
  torch.cuda.synchronize();captured=[]
  for i in range(12):
   torch.manual_seed(197743+n+i*100003);pairs[...,0].normal_();pairs[...,1].copy_(torch.randint(0,2**24,(8,n),device='cuda').float())
   row={k:[] for k in funcs}
   for name in ('native','candidate','control','control','candidate','native'):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    for _ in range(16):graphs[name].replay()
    end.record();end.synchronize();row[name].append(start.elapsed_time(end)*1000/(128*16))
   assert torch.equal(outputs['native'],outputs['candidate']) and torch.equal(outputs['native'],outputs['control'])
   captured.append(row)
  eager=[]
  for _ in range(20):
   for fn in funcs.values():fn(pairs)
  torch.cuda.synchronize()
  for i in range(6):
   row={k:[] for k in funcs}
   for name in ('native','candidate','control','control','candidate','native'):
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(512):out=funcs[name](pairs)
    torch.cuda.synchronize();row[name].append((time.perf_counter()-start)*1e6/512)
    assert torch.equal(out,native_select(pairs))
   eager.append(row)
  result=dict(n=n,exact=True,captured=dict(records=captured,summary=summary(captured)),eager=dict(records=eager,summary=summary(eager)))
  report['cases'].append(result);save();print(json.dumps(dict(n=n,captured=result['captured']['summary'],eager=result['eager']['summary'])),flush=True)
  del graphs,outputs,g,out,pairs;gc.collect();torch.cuda.empty_cache()
 report.update(complete=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),serving_qualified=False);save()
if __name__=='__main__':main()
