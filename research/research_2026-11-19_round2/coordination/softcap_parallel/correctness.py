import pathlib,resource,importlib.util,hashlib,json,gc
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
mem=dict((x.split(':')[0],int(x.split()[1])) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:')))
if mem['MemAvailable']<2*1024**2:raise RuntimeError('Host memory preflight below2GiB')
import torch,triton
from parallel_softcap import parallel_softcap,_partial,_finish
HERE=pathlib.Path(__file__).resolve().parent
source=HERE.parents[2]/'sglang/python/sglang/kernels/ops/sampling/softcap_row_argmax.py'
spec=importlib.util.spec_from_file_location('native_softcap',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report=dict(complete=False,scope='SM86 native4096-block softcap payload equivalence, noGEMM/collective',native_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((HERE/'parallel_softcap.py').read_bytes()).hexdigest(),host_memory_kib=mem,cases=[])
 def save():(HERE/'correctness.json').write_text(json.dumps(report,indent=2)+'\n')
 save()
 for n in (1,8,32,128,256):
  for width in (1,4095,4096,4097,25256,32768,65536):
   backing=torch.empty((n,width+19),device='cuda',dtype=torch.float32);logits=backing[:,:width]
   for valid in sorted(set((0,max(0,width-13),width))):
    tiles=max(1,triton.cdiv(valid,4096));scratch_storage=torch.full((n*tiles*2+32,),12345.0,device='cuda');scratch=scratch_storage[16:-16]
    out_storage=torch.full((n*2+32,),12345.0,device='cuda');out=out_storage[16:-16].view(n,2)
    offset=7*width
    for cap in (1.0,20.0,30.0,50.0):
     def launch():
      _partial[(n,tiles)](logits,scratch,logits.stride(0),valid,cap,tiles,4096,num_warps=8)
      _finish[(n,)](scratch,out,offset,tiles,num_warps=4)
     logits.normal_();launch();ref=native.softcap_row_argmax(logits,cap,valid,offset);torch.cuda.synchronize()
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):ref=native.softcap_row_argmax(logits,cap,valid,offset);launch()
     for pattern in range(7):
      torch.manual_seed(731879+n*31+width*13+valid+pattern*997);logits.normal_()
      if pattern==1:logits.round_()
      elif pattern==2:
       logits.mul_(1e6)
       if width>4096:logits[:,4095:4097]=float('inf')
      elif pattern==3:
       logits[:,::4096]=float('nan')
       if width>4096:logits[:,4096:8192]=float('nan')
      elif pattern==4:logits.fill_(float('nan'))
      elif pattern==5:
       raw=torch.tensor([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f7fffff,0xff7fffff],device='cuda',dtype=torch.int64).to(torch.int32)
       logits.view(torch.int32).copy_(raw[torch.arange(n*width,device='cuda')%raw.numel()].view(n,width))
      elif pattern==6:
       logits.fill_(-float('inf'))
       if valid:logits[:,valid-1]=0.0
      graph.replay();torch.cuda.synchronize()
      exact=torch.equal(ref.view(torch.int32),out.view(torch.int32))
      guards=bool(torch.all(scratch_storage[:16]==12345) and torch.all(scratch_storage[-16:]==12345) and torch.all(out_storage[:16]==12345) and torch.all(out_storage[-16:]==12345))
      case=dict(n=n,width=width,valid=valid,cap=cap,pattern=pattern,exact=exact,guards=guards)
      report['cases'].append(case)
      if not exact or not guards:
       report['failure']=dict(case=case,reference=ref.cpu().tolist(),candidate=out.cpu().tolist());save();raise SystemExit(2)
     del graph,ref
    del scratch,scratch_storage,out,out_storage
   del logits,backing;gc.collect();torch.cuda.empty_cache();save()
  print('N',n,'exact',len(report['cases']),flush=True)
 report.update(complete=True,all_exact=True,count=len(report['cases']),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save();print('PASS',report['count'],flush=True)
if __name__=='__main__':main()
