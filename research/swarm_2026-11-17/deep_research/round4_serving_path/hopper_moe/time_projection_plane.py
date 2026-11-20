"""Bounded local projection producer timing; no communication proxy."""
import argparse,hashlib,json,pathlib,statistics,subprocess,time
P=pathlib.Path(__file__).resolve().parent

def main():
 import torch
 import torch.nn.functional as F
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=P/a.out;out.mkdir(exist_ok=False)
 assert torch.cuda.get_device_capability()==(8,6)
 free,total=torch.cuda.mem_get_info();assert free>=512*1024**2+384*1024**2
 torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();start=time.monotonic();deadline=start+90
 result=dict(status='RUNNING',scope='SM86 local BF16 projection producer plus optionalcopy; noNCCLproxy/noH200forecast',source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),rows=[],initial_free_bytes=free)
 def telemetry():
  try:return subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem','--format=csv,noheader'],text=True,timeout=3).strip()
  except Exception as e:return str(e)
 result['telemetry_start']=telemetry()
 def save():(out/'results.json').write_text(json.dumps(result,indent=2))
 save()
 def sample(g,repeats=10):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(repeats):g.replay()
  b.record();b.synchronize();return a.elapsed_time(b)*1000/repeats
 def ratio(a,b):return statistics.median(a)/statistics.median(b)
 def drift(v):return ratio(v[:len(v)//2],v[len(v)//2:])
 for m in [32,128,256,753]:
  assert time.monotonic()<deadline
  torch.manual_seed(17700+m);x=torch.randn((m,1024),device='cuda',dtype=torch.bfloat16);w=torch.randn((8192,1024),device='cuda',dtype=torch.bfloat16)*.125
  planes=[torch.full((m*8192+512,),37,device='cuda',dtype=torch.bfloat16)for _ in range(4)];views=[p[256:-256].view(m,8192)for p in planes]
  def run(i):
   if i in [0,3]:views[i].copy_(F.linear(x,w,None))
   else:torch.mm(x,w.t(),out=views[i])
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for i in range(4):
    for _ in range(3):run(i)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  graphs=[]
  for i in range(4):
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):run(i)
   graphs.append(g)
  for i,g in enumerate(graphs):
   views[i].fill_(float('nan'));g.replay()
  torch.cuda.synchronize()
  assert all(torch.isfinite(v).all().item() and torch.equal(v.view(torch.int16),views[0].view(torch.int16))for v in views)
  assert all(torch.equal(p[:256],torch.full_like(p[:256],37)) and torch.equal(p[-256:],torch.full_like(p[-256:],37))for p in planes)
  # Time-based warmup per graph to avoid first-sample clock/cache ramp artifacts.
  for g in graphs:
   until=time.monotonic()+.3
   while time.monotonic()<until:
    for _ in range(10):g.replay()
    torch.cuda.synchronize()
  opening=[[],[]]
  for _ in range(8):opening[0].append(sample(graphs[0]));opening[1].append(sample(graphs[3]))
  samples=[[]for _ in range(4)]
  for _ in range(8):
   for i in [0,1,2,3]:samples[i].append(sample(graphs[i]))
  closing=[[],[]]
  for _ in range(8):closing[0].append(sample(graphs[0]));closing[1].append(sample(graphs[3]))
  checks=dict(opening_AA=ratio(*opening),closing_AA=ratio(*closing),AA=ratio(samples[0],samples[3]),BB=ratio(samples[1],samples[2]),opening_closing_A=ratio(opening[0]+opening[1],closing[0]+closing[1]),drift=[drift(v)for v in samples])
  qualified=all(.97<=checks[k]<=1.03 for k in ['opening_AA','closing_AA','AA','BB']) and .95<=checks['opening_closing_A']<=1.05 and all(.95<=v<=1.05 for v in checks['drift'])
  baseline=statistics.median(samples[0]+samples[3]);direct=statistics.median(samples[1]+samples[2])
  result['rows'].append(dict(M=m,N=8192,K=1024,baseline_linear_copy_us=baseline,direct_mm_out_us=direct,saved_us=baseline-direct,qualified=qualified,controls=checks,samples_us=samples,opening_us=opening,closing_us=closing,telemetry=telemetry()))
  save();del graphs,g,x,w,planes,views
 result.update(status='COMPLETE_LOCAL_PRODUCER_TIMING',seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),telemetry_end=telemetry(),no_serving_speedup_claim=True);save();print(json.dumps(result,indent=2))
if __name__=='__main__':main()
