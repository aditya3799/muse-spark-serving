"""Native C1 receive+norm arithmetic under single-GPU virtual peers, not NVLink."""
import ctypes as ct,torch,json,hashlib,pathlib,statistics,math,gc,argparse
HERE=pathlib.Path(__file__).resolve().parent
class Params(ct.Structure):
 _fields_=[(n,ct.c_void_p) for n in ('branch','stream','output','new_stream','router_out','post_weight','pre_weight','alpha','beta')]+[('post_eps',ct.c_float),('pre_eps',ct.c_float),('num_rows',ct.c_int),('gate_channel_wise',ct.c_bool),('has_pre_weight',ct.c_bool),('pre_norm_bf16',ct.c_bool)]
lib=ct.CDLL(str(HERE/'sim_sm86.so'));V=ct.c_void_p;I=ct.c_int
lib.norm_params_size.restype=ct.c_size_t;lib.norm_offset.argtypes=[I];lib.norm_offset.restype=ct.c_size_t
assert lib.norm_params_size()==ct.sizeof(Params)
assert all(lib.norm_offset(i)==getattr(Params,n).offset for i,(n,_) in enumerate(Params._fields_))
lib.pack_sim.argtypes=[V,V,I,V];lib.gather_sim.argtypes=[V,V,I,I,V];lib.norm_sim.argtypes=[I,I,I,I,I,V,V,V]
for fn in (lib.pack_sim,lib.gather_sim,lib.norm_sim):fn.restype=I

def checked(rc):
 if rc:raise RuntimeError('CUDA status '+str(rc))

def ptr(x):return None if x is None else x.data_ptr()

@torch.no_grad()
def main(seed):
 torch.set_num_threads(2);torch.manual_seed(seed);stream=torch.cuda.current_stream().cuda_stream
 cap=torch.cuda.get_device_properties(0).multi_processor_count
 report={'scope':__doc__,'seed':seed,'device':torch.cuda.get_device_name(),'abi_size':ct.sizeof(Params),'abi_offsets_checked':15,'source_sha256':hashlib.sha256((HERE/'sim_native.cu').read_bytes()).hexdigest(),'cases':[],'complete':False,'distributed_execution':False,'serving_qualified':False}
 path=HERE/('probe_'+str(seed)+'.json')
 def save():path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for m in (1,8,32,128,192,256,512):
  for scale in (False,True):
   threads=256 if m<192 else 512
   x=torch.empty((m,8192),device='cuda',dtype=torch.bfloat16)
   residual=torch.empty((m,8192),device='cuda',dtype=torch.float32)
   post=torch.empty(8192,device='cuda',dtype=torch.bfloat16)
   pre=torch.empty_like(post)
   alpha=torch.empty(8192 if scale else 1,device='cuda');beta=torch.empty_like(alpha)
   arms={};outputs={};graphs={}
   for name in ('native','candidate','control'):
    storage=torch.full((m*8192+32,),7.,device='cuda',dtype=torch.bfloat16);packed=storage[16:-16].view(m,8192)
    branch=torch.empty_like(x);out=torch.empty_like(x);new=torch.empty_like(residual)
    params=Params(ptr(branch),ptr(residual),ptr(out),ptr(new),None,ptr(post),ptr(pre),ptr(alpha),ptr(beta),1e-6,1e-6,m,scale,True,False)
    arms[name]=(storage,packed,branch,params);outputs[name]=(out,new)
   def run(name,gate=True):
    stream=torch.cuda.current_stream().cuda_stream
    _,packed,branch,params=arms[name]
    checked(lib.pack_sim(ptr(x),ptr(packed),m,stream))
    if name!='candidate':checked(lib.gather_sim(ptr(packed),ptr(branch),m,cap,stream))
    checked(lib.norm_sim(name=='candidate',scale,gate,threads,cap,ptr(packed),ct.byref(params),stream))
   def update(iteration):
    torch.manual_seed(seed+iteration*8191+m+int(scale))
    x.normal_();residual.normal_();post.uniform_(.5,1.5);pre.uniform_(.5,1.5);alpha.uniform_(.7,1.3);beta.uniform_(.7,1.3)
    if iteration%4==1:x[:,::17]=0
    if iteration%4==2:x.mul_(128);residual.mul_(.01)
    if iteration%4==3:x.copy_(torch.randint(-4,5,x.shape,device='cuda').to(x.dtype))
   update(0)
   for name in arms:
    run(name);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for _ in range(32):run(name)
    graphs[name]=graph
   exact=True;records=[];times={n:[] for n in arms}
   for iteration in range(8):
    update(iteration)
    order=('native','candidate','control') if iteration%2==0 else ('control','candidate','native')
    for name in order:
     begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
     begin.record();graphs[name].replay();end.record();end.synchronize();times[name].append(begin.elapsed_time(end)*1000/32)
    equal=all(torch.equal(outputs['native'][j].view(torch.int16 if j==0 else torch.int32),outputs['candidate'][j].view(torch.int16 if j==0 else torch.int32)) for j in (0,1))
    control=all(torch.equal(outputs['native'][j].view(torch.int16 if j==0 else torch.int32),outputs['control'][j].view(torch.int16 if j==0 else torch.int32)) for j in (0,1))
    canary=all(bool((a[0][:16]==7).all() and (a[0][-16:]==7).all()) for a in arms.values())
    cleared=all(int(torch.count_nonzero(a[1].view(torch.int16)))==0 for a in arms.values())
    exact=exact and equal and control and canary and cleared
    records.append(dict(iteration=iteration,output_exact=equal,control_exact=control,canary=canary,receive_cleared=cleared))
   # Independently check full rank-major packing and signed-zero convention.
   storage,packed,branch,params=arms['candidate'];checked(lib.pack_sim(ptr(x),ptr(packed),m,stream))
   expected=x.clone();expected.view(torch.int16)[expected.view(torch.int16)==0]=-32768
   encoded=packed.view(8,m,1024).permute(1,0,2).reshape(m,8192)
   packing_exact=torch.equal(encoded.view(torch.int16),expected.view(torch.int16));exact &= packing_exact
   # Both gating variants execute even though the timed fixture uses the model gate.
   for name in arms:run(name,False)
   ungated=all(torch.equal(outputs['native'][j].view(torch.int16 if j==0 else torch.int32),outputs['candidate'][j].view(torch.int16 if j==0 else torch.int32)) for j in (0,1));exact &= ungated
   row=dict(m=m,scale=scale,threads=threads,counter_cap=cap,exact=exact,records=records,packing_exact=packing_exact,ungated_exact=ungated,times_us=times,gain=statistics.median(times['native'])/statistics.median(times['candidate']),control_ratio=statistics.median(times['native'])/statistics.median(times['control']))
   report['cases'].append(row);save();print(m,scale,exact,round(row['gain'],4),flush=True)
   del graphs,arms,outputs,storage,packed,branch,params,out,new,graph,x,residual,post,pre,alpha,beta,expected,encoded
   gc.collect();torch.cuda.empty_cache()
 report['complete']=True;report['all_exact']=all(c['exact'] for c in report['cases']);report['peak_allocated']=torch.cuda.max_memory_allocated();report['peak_reserved']=torch.cuda.max_memory_reserved();save()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=419731);a=parser.parse_args();main(a.seed)
