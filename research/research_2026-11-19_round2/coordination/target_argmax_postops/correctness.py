import pathlib,resource,json,hashlib,importlib.util,sys,gc
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
mem=dict((x.split(':')[0],int(x.split()[1])) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:')))
if mem['MemAvailable']<2*1024**2:raise RuntimeError('Host memory preflight below 2 GiB')
import torch,triton
from target_select import select_target,native_select,fused_select
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
path=ROOT/'sglang/python/sglang/kernels/ops/sampling/softcap_row_argmax.py'
spec=importlib.util.spec_from_file_location('native_target_softcap',path);softcap=importlib.util.module_from_spec(spec);spec.loader.exec_module(softcap)
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report=dict(scope='Local target post-gather operations; native float32 payload and conversion preserved; no transport timing',source_sha256=hashlib.sha256((HERE/'target_select.py').read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),host_memory_kib=mem,complete=False,cases=[],producer_cases=[])
 outpath=HERE/'correctness.json'
 def save():outpath.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for tp in (1,2,3,4,5,8,16):
  for n in (1,7,8,31,32,56,128,224,256,257,1024):
   pairs=torch.empty((tp,n,2),device='cuda',dtype=torch.float32)
   storage=torch.full((n+32,),0x17325566,device='cuda',dtype=torch.int64);out=storage[16:-16]
   def launch():select_target[(triton.cdiv(n,32),)](pairs,out,n,tp,triton.next_power_of_2(tp),32,num_warps=4)
   pairs.normal_();pairs[...,1].copy_(torch.arange(tp*n,device='cuda').view(tp,n))
   launch();native_select(pairs);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):reference=native_select(pairs);launch()
   records=[]
   for pattern in range(7):
    torch.manual_seed(197743+tp*173+n*8191+pattern*100003)
    pairs.normal_();pairs[...,1].copy_(torch.randint(0,2**24,(tp,n),device='cuda').float())
    if pattern==1:pairs[...,0].copy_(torch.randint(-2,3,(tp,n),device='cuda').float())
    elif pattern==2:pairs[...,0].zero_();pairs[::2,:,0]=-0.0
    elif pattern==3:pairs[...,0].fill_(float('-inf'));pairs[-1,:,0]=float('inf')
    elif pattern==4:
     pairs[::2,:,0]=float('nan')
     if tp>1:pairs[-1,:,0]=float('nan')
    elif pattern==5:
     values=pairs[...,0].contiguous();bits=values.view(torch.int32);raw=torch.tensor([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc12345,0x7f7fffff,0xff7fffff],device='cuda',dtype=torch.int64).to(torch.int32);bits.copy_(raw[torch.arange(tp*n,device='cuda')%raw.numel()].view(tp,n));pairs[...,0].copy_(values)
    elif pattern==6:
     ids=torch.tensor([0.,-0.,1.75,-1.75,16777217.,-9223372036854775808.,9223372036854775808.,float('inf'),-float('inf'),float('nan'),3.402823466e38,-3.402823466e38],device='cuda',dtype=torch.float32)
     pairs[...,1].copy_(ids[torch.arange(tp*n,device='cuda')%ids.numel()].view(tp,n))
    graph.replay();torch.cuda.synchronize()
    exact=torch.equal(reference,out);guards=bool(torch.all(storage[:16]==0x17325566) and torch.all(storage[-16:]==0x17325566))
    records.append(dict(pattern=pattern,exact=exact,guards=guards))
    if not exact or not guards:
     report['failure']=dict(tp=tp,n=n,pattern=pattern,reference=reference.cpu().tolist(),candidate=out.cpu().tolist());save();raise SystemExit(2)
   report['cases'].append(dict(tp=tp,n=n,records=records));save()
   del graph,reference,out,storage,pairs
  print('TP',tp,'payloads exact',flush=True)
 for tp in (2,8):
  for n in (8,128):
   for width in (513,4099):
    logits=torch.empty((tp,n,width),device='cuda',dtype=torch.float32)
    for style in ('normal','ties','extreme'):
     torch.manual_seed(673193+tp+n+width);logits.normal_()
     if style=='ties':logits.round_()
     if style=='extreme':logits.mul_(1e6);logits[:,:,::127]=float('nan')
     local=[softcap.softcap_row_argmax(logits[r],30.0,width if r<tp-1 else width-13,r*width) for r in range(tp)]
     pairs=torch.stack(local);ref=native_select(pairs);got=fused_select(pairs)
     exact=torch.equal(ref,got);report['producer_cases'].append(dict(tp=tp,n=n,width=width,style=style,exact=exact));save()
     if not exact:raise SystemExit(3)
    del logits,local,pairs,ref,got
 report.update(complete=True,all_exact=True,payload_cases=sum(len(c['records']) for c in report['cases']),native_producer_cases=len(report['producer_cases']),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save();print('all exact',report['payload_cases'],report['native_producer_cases'],flush=True)
if __name__=='__main__':main()
