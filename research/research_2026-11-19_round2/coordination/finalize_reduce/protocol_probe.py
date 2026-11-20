import ctypes as ct,torch,pathlib,json,hashlib,sys,importlib.util,gc,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
base=ROOT/'release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_post_expert_finalize_triton.py'
spec=importlib.util.spec_from_file_location('frozen_protocol_finalize',base);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
class Args(ct.Structure):
 _fields_=[(n,ct.c_void_p) for n in ('x','weight','scale','peer','local','scores','inv','ids','out','debug_local')]+[(n,ct.c_int) for n in ('rows','hidden','rank')]
class Proto(ct.Structure):
 _fields_=[('ranks',Args*8),('workspaces',ct.c_void_p*8),('counters',ct.c_void_p*8),('slot_bytes',ct.c_uint32),('controllers',ct.c_uint32)]
lib=ct.CDLL(str(HERE/'protocol_sm86.so'));lib.proto_args_size.restype=ct.c_size_t;assert lib.proto_args_size()==ct.sizeof(Proto)
lib.run_protocol.argtypes=[ct.c_int,ct.c_int,ct.POINTER(Proto),ct.c_void_p];lib.run_protocol.restype=ct.c_int

def main():
 torch.set_num_threads(2)
 mem=dict((x.split(':')[0],int(x.split()[1])) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:')))
 if mem['MemAvailable']<2*1024**2:raise RuntimeError('Host memory preflight below 2 GiB')
 report=dict(scope='Single GPU virtual eight-rank exact native push/poll/clear/epoch protocol. Not NVLink, IPC, NCCL or multi-device validation; no timing claim.',source_sha256=hashlib.sha256((HERE/'protocol.cu').read_bytes()).hexdigest(),abi_size=ct.sizeof(Proto),host_memory_kib=mem,complete=False,cases=[])
 path=HERE/'protocol_probe.json'
 def save():path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for controllers in (1,2):
  for h in (1024,4096,8192):
   slot_bytes=8*h*2;plane_bytes=16*slot_bytes
   plane_storage=torch.full((8,plane_bytes+64),167,device='cuda',dtype=torch.uint8)
   planes=[plane_storage[r,32:-32] for r in range(8)]
   for plane in planes:plane.zero_()
   counter_storage=torch.full((8,controllers+8),0x5a5a5a5a,device='cuda',dtype=torch.int32)
   counters=[counter_storage[r,4:-4] for r in range(8)]
   for counter in counters:counter.zero_()
   for m in (2,8,3):
    for scale in (False,True):
     x=torch.empty((8,m,8,h),device='cuda',dtype=torch.bfloat16);weight=torch.empty(h,device='cuda',dtype=torch.bfloat16);scales=torch.empty((8,32),device='cuda',dtype=torch.bfloat16)
     scores=torch.empty((8,m,8),device='cuda');ids=torch.empty((8,m,8),device='cuda',dtype=torch.int32);inv=torch.empty_like(scores);local=torch.empty((8,m,h),device='cuda',dtype=torch.bfloat16);debug=torch.empty_like(local)
     outstorage=torch.full((2,8,m*h+32),-123,device='cuda',dtype=torch.bfloat16)
     outs=[[outstorage[a,r,16:-16].view(m,h) for r in range(8)] for a in range(2)]
     params=[]
     for arm in range(2):
      pp=Proto();pp.slot_bytes=slot_bytes;pp.controllers=controllers
      for r in range(8):
       pp.ranks[r]=Args(x[r].data_ptr(),weight.data_ptr(),scales[r].data_ptr(),None,local[r].data_ptr(),scores[r].data_ptr(),inv[r].data_ptr(),ids[r].data_ptr(),outs[arm][r].data_ptr(),debug[r].data_ptr() if arm else None,m,h,r)
       pp.workspaces[r]=planes[r].data_ptr();pp.counters[r]=counters[r].data_ptr()
      params.append(pp)
     def rms_combine():
      for r in range(8):
       native._post_expert_inv_rms_kernel[(m*8,)](x[r],weight,ids[r],inv[r],scales[r],HAS_SCALE=scale,TOPK=8,H=h,eps=1e-8,HAS_WEIGHT=True,WEIGHT_FIRST=True,BLOCK_H=h,num_warps=8 if h>=4096 else 4,num_stages=2)
       native._post_expert_tiled_combine_kernel[(m,h//256)](x[r],weight,scores[r],ids[r],inv[r],local[r],scales[r],HAS_SCALE=scale,TOPK=8,H=h,HAS_WEIGHT=True,WEIGHT_FIRST=True,BLOCK=256,num_warps=4)
     def launch(arm):
      rc=lib.run_protocol(arm,scale,ct.byref(params[arm]),torch.cuda.current_stream().cuda_stream)
      if rc:raise RuntimeError('protocol launch status '+str(rc))
     def update(i):
      torch.manual_seed(671237+i*100003+h+m*8191+controllers*191)
      x.normal_();weight.uniform_(.5,1.5);scales.uniform_(.5,1.5);scores.uniform_(0,.3);ids.random_(-5,32)
      if i==1:ids.fill_(-1);x.fill_(float('nan'))
      elif i==2:ids[:,:,::2]=-1;x[:,:,::2]=float('nan');x[::2].mul_(128);x[1::2].mul_(.001)
      elif i==3:x.zero_();scores.zero_()
      elif i==4:x[0].fill_(128);x[1].fill_(-128);x[2].fill_(.001)
     update(0);rms_combine();launch(0);launch(1);torch.cuda.synchronize()
     graphs=[]
     for arm in range(2):
      g=torch.cuda.CUDAGraph()
      with torch.cuda.graph(g):launch(arm)
      graphs.append(g)
     records=[]
     for i in range(5):
      update(i);rms_combine()
      encoded=local.clone();words=encoded.view(torch.int32);words[words==0]=0x8000
      total=encoded[0].float()
      for r in range(1,8):total=total+encoded[r].float()
      expected=total.to(torch.bfloat16)
      before=counters[0].clone()
      # Changed data and interleaving of native/fused callers on the same plane.
      for arm in (0,1,0,1,1):graphs[arm].replay()
      torch.cuda.synchronize()
      eq=all(torch.equal(out.view(torch.int16),expected.view(torch.int16)) for group in outs for out in group)
      local_eq=torch.equal(debug.view(torch.int16),local.view(torch.int16))
      guards=bool(torch.all(plane_storage[:,:32]==167) and torch.all(plane_storage[:,-32:]==167) and torch.all(counter_storage[:,:4]==0x5a5a5a5a) and torch.all(counter_storage[:,-4:]==0x5a5a5a5a) and torch.all(outstorage[:,:,:16]==-123) and torch.all(outstorage[:,:,-16:]==-123))
      clear=all(torch.count_nonzero(v).item()==0 for v in planes)
      epochs=all(torch.equal(v,before^1) for v in counters)
      row=dict(pattern=i,output_exact=eq,local_exact=local_eq,guards_exact=guards,both_planes_cleared=clear,all_counters_toggled=epochs)
      records.append(row)
      if not all((eq,local_eq,guards,clear,epochs)):
       report.update(failure=dict(m=m,h=h,scale=scale,controllers=controllers,record=row));save();raise SystemExit(2)
     report['cases'].append(dict(m=m,h=h,scale=scale,controllers=controllers,records=records));save();print(m,h,scale,controllers,'exact',flush=True)
     del graphs,g,params,outs,outstorage,x,weight,scales,scores,ids,inv,local,debug,encoded,words,total,expected,before
     gc.collect()
   del planes,counters,plane_storage,counter_storage
   torch.cuda.empty_cache()
 report.update(complete=True,all_exact=True,configurations=len(report['cases']),changed_input_cases=5*len(report['cases']),rank_output_cases=40*len(report['cases']),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved());save()
if __name__=='__main__':main()
