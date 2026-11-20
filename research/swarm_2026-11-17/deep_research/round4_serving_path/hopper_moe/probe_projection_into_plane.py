"""Local BF16 F.linear+copy vs mm.out-to-plane producer gate. No collectives."""
import argparse,hashlib,json,pathlib,time
P=pathlib.Path(__file__).resolve().parent

def main():
 import torch
 import torch.nn.functional as F
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=P/a.out;out.mkdir(exist_ok=False)
 assert torch.cuda.get_device_capability()==(8,6),'local diagnostic guard'
 free,total=torch.cuda.mem_get_info();assert free>=512*1024**2+192*1024**2,'512MiB reserve'
 torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();start=time.time()
 report=dict(status='STARTED',scope='unquantized torch backend producer only; no MC or full model',checks=[],shape_source='pinned config defaults128Qheads*64head_dim/TP8 =1024; corroborated cached challenge reference',source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),torch_version=torch.__version__,cuda=torch.version.cuda,matmul_flags=dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32,allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),initial_free_bytes=free)
 def save(): (out/'results.json').write_text(json.dumps(report,indent=2))
 save()
 for m in [32,128,256,753]:
  x=torch.empty((m,1024),device='cuda',dtype=torch.bfloat16);w=torch.empty((8192,1024),device='cuda',dtype=torch.bfloat16)
  planes=[torch.full((m*8192+512,),37,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
  views=[p[256:-256].view(m,8192) for p in planes]
  assert views[0].data_ptr()!=views[1].data_ptr()
  assert all(v.is_contiguous() and v.data_ptr()%256==0 for v in views)
  def baseline():views[0].copy_(F.linear(x,w,None))
  def direct():torch.mm(x,w.t(),out=views[1])
  def populate(gen):
   torch.manual_seed(500+m*11+gen)
   x.copy_(torch.randn_like(x));w.copy_(torch.randn_like(w)*.125)
   if gen==1:
    x[:,::2]=1;x[:,1::2]=-1
   if gen==2:
    x.zero_();x[:,0]=2**12;x[:,32]=1;x[:,64]=-(2**12);x[:,96]=1
    w.fill_(1)
  def check(gen,mode):
   torch.cuda.synchronize()
   assert torch.isfinite(views[0]).all().item() and torch.isfinite(views[1]).all().item()
   equal=torch.equal(views[0].view(torch.int16),views[1].view(torch.int16))
   sentinel=all(torch.equal(p[:256],torch.full_like(p[:256],37)) and torch.equal(p[-256:],torch.full_like(p[-256:],37)) for p in planes)
   item=dict(M=m,K=1024,N=8192,generation=gen,mode=mode,bitwise=equal,finite=True,guards=sentinel,max_abs_error=(views[0].float()-views[1].float()).abs().max().item())
   report['checks'].append(item);save();assert equal and sentinel,item
  for gen in range(3):
   populate(gen)
   for v in views:v.fill_(float('nan'))
   baseline();direct();check(gen,'eager')
  populate(3)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3):baseline();direct()
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
  ga=torch.cuda.CUDAGraph();gb=torch.cuda.CUDAGraph()
  with torch.cuda.graph(ga):baseline()
  with torch.cuda.graph(gb):direct()
  for gen in [4,5]:
   populate(gen)
   for v in views:v.fill_(float('nan'))
   ga.replay();gb.replay();check(gen,'changed_pointer_stable_graph')
  del ga,gb,x,w,views,planes
 report.update(status='PASS_LOCAL_PRODUCER_ONLY',peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,claims=dict(multigpu=False,H200=False,serving_gain=False,backend_expansion=False),remaining=['native H200 same-backend numeric check','exact MC plane pointer/IPC safety','full attention norm-chain/captured replay across communicator generations','no GEMV/custom/deterministic mode admission'])
 save();print(json.dumps(report,indent=2))
if __name__=='__main__':main()
