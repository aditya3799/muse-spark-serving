import ctypes as ct,torch,triton,pathlib,importlib.util,json,hashlib,statistics,gc,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
base=ROOT/'release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
spec=importlib.util.spec_from_file_location('frozen_finalize_exact',base);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
class Args(ct.Structure):
 _fields_=[(n,ct.c_void_p) for n in ('x','weight','scale','peer','local','scores','inv','ids','out','debug_local')]+[(n,ct.c_int) for n in ('rows','hidden','rank')]
lib=ct.CDLL(str(HERE/'combine_sm86.so'));lib.args_size.restype=ct.c_size_t;assert lib.args_size()==ct.sizeof(Args)
lib.run.argtypes=[ct.c_int]*4+[ct.POINTER(Args),ct.c_void_p];lib.run.restype=ct.c_int

def run_cuda(fused,w,f,s,a):
 rc=lib.run(fused,w,f,s,ct.byref(a),torch.cuda.current_stream().cuda_stream)
 if rc:raise RuntimeError('CUDA status '+str(rc))

@torch.no_grad()
def main():
 torch.set_num_threads(2)
 report={'scope':'Original inverseRMS + nativeTriton tiledcombine + local virtual-peer reduction versus CUDA combine/reduction; NO real transport','source_sha256':hashlib.sha256((HERE/'combine_reduce.cu').read_bytes()).hexdigest(),'parent_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'abi_size':ct.sizeof(Args),'complete':False,'cases':[]}
 path=HERE/'probe.json'
 def save():path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for h in (1024,4096,8192):
  for m in (2,3,8):
   for rank,w,first,scale in ((0,True,True,False),(3,True,False,True),(7,False,False,True)):
    x=torch.empty((m,8,h),device='cuda',dtype=torch.bfloat16);weight=torch.empty(h,device='cuda',dtype=torch.bfloat16);scales=torch.empty(32,device='cuda',dtype=torch.bfloat16)
    scores=torch.empty((m,8),device='cuda');ids=torch.empty((m,8),device='cuda',dtype=torch.int32);peers=torch.empty((8,m,h),device='cuda',dtype=torch.bfloat16)
    inv=torch.empty((m,8),device='cuda');local=torch.empty((m,h),device='cuda',dtype=torch.bfloat16);debug=torch.empty_like(local)
    outputs={n:torch.empty_like(local) for n in ('native','candidate','control')}
    params={n:Args(x.data_ptr(),weight.data_ptr(),scales.data_ptr(),peers.data_ptr(),local.data_ptr(),scores.data_ptr(),inv.data_ptr(),ids.data_ptr(),out.data_ptr(),None,m,h,rank) for n,out in outputs.items()}
    def rms():
     native._post_expert_inv_rms_kernel[(m*8,)](x,weight,ids,inv,scales,HAS_SCALE=scale,TOPK=8,H=h,eps=1e-8,HAS_WEIGHT=w,WEIGHT_FIRST=first,BLOCK_H=h,num_warps=8 if h>=4096 else 4,num_stages=2)
    def combine():
     native._post_expert_tiled_combine_kernel[(m,h//256)](x,weight,scores,ids,inv,local,scales,HAS_SCALE=scale,TOPK=8,H=h,HAS_WEIGHT=w,WEIGHT_FIRST=first,BLOCK=256,num_warps=4)
    def run(name):
     rms()
     if name!='candidate':combine()
     run_cuda(name=='candidate',w,first,scale,params[name])
    records=[];exact=True
    def update(i):
     torch.manual_seed(539143+i*100003+h+m*8191+rank)
     x.normal_();weight.uniform_(.5,1.5);scales.uniform_(.5,1.5);scores.uniform_(0,.3);ids.random_(-5,32);peers.normal_()
     if i==1:
      ids.fill_(-1);x.fill_(float('nan'));peers.zero_()
     elif i==2:
      ids[:,::2]=-1;x[:,::2]=float('nan');peers[::2].mul_(128);peers[1::2].mul_(.001)
     elif i==3:
      x.zero_();peers.zero_();scores.zero_()
     elif i==4:
      x.mul_(128);peers[0].fill_(128);peers[1].fill_(-128);peers[2].fill_(.001)
    update(0)
    for n in params:run(n)
    torch.cuda.synchronize()
    graphs={}
    for n in params:
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):
      for _ in range(16):run(n)
     graphs[n]=g
    times={n:[] for n in params}
    for i in range(5):
     update(i)
     for n in ('native','candidate','control') if i%2==0 else ('control','candidate','native'):
      b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);b.record();graphs[n].replay();e.record();e.synchronize();times[n].append(b.elapsed_time(e)*1000/16)
     eq=torch.equal(outputs['native'].view(torch.int16),outputs['candidate'].view(torch.int16));ctrl=torch.equal(outputs['native'].view(torch.int16),outputs['control'].view(torch.int16))
     # Compare the local BF16 combine before encoding, with the same inverseRMS.
     rms();combine();params['candidate'].debug_local=debug.data_ptr();run_cuda(True,w,first,scale,params['candidate']);params['candidate'].debug_local=None
     local_eq=torch.equal(local.view(torch.int16),debug.view(torch.int16))
     # Independent increasing-rank sum and exact zero-pair encoding.
     incoming=peers.clone();incoming[rank].copy_(local);words=incoming.view(torch.int32);words[words==0]=0x8000
     total=incoming[0].float()
     for r in range(1,8):total=total+incoming[r].float()
     oracle=torch.equal(total.to(torch.bfloat16).view(torch.int16),outputs['native'].view(torch.int16))
     exact &=eq and ctrl and local_eq and oracle
     records.append({'case':i,'output_exact':eq,'local_combine_exact':local_eq,'control_exact':ctrl,'independent_reduce_exact':oracle})
    row={'m':m,'h':h,'rank':rank,'weight':w,'weight_first':first,'scale':scale,'exact':exact,'records':records,'times_us':times,'gain':statistics.median(times['native'])/statistics.median(times['candidate']),'control_ratio':statistics.median(times['native'])/statistics.median(times['control'])}
    report['cases'].append(row);save();print(m,h,rank,exact,round(row['gain'],4),flush=True)
    if not exact:
     report['stopped_on_mismatch']=True;save();raise SystemExit(2)
    del graphs,g,params,outputs,x,weight,scales,scores,ids,peers,inv,local,debug,incoming,words,total
    gc.collect();torch.cuda.empty_cache()
 report.update(complete=True,all_exact=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save()
if __name__=='__main__':main()
