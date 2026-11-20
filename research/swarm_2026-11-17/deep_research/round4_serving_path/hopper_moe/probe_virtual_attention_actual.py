"""Bounded virtual-rank AR/norm gate. Requires explicit root GPU/compiler lease."""
import argparse,ctypes,hashlib,json,os,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
class Norm(ctypes.Structure):
 _fields_=[(n,ctypes.c_void_p) for n in ['branch','stream','output','new_stream','router_out','post_weight','pre_weight','alpha','beta']]+[('post_eps',ctypes.c_float),('pre_eps',ctypes.c_float),('num_rows',ctypes.c_int),('gate_channel_wise',ctypes.c_bool),('has_pre_weight',ctypes.c_bool),('pre_norm_bf16',ctypes.c_bool)]
class WS(ctypes.Structure):
 _fields_=[('workspaces',ctypes.c_void_p*8),('counter',ctypes.c_void_p),('mc_workspace',ctypes.c_void_p),('slot_bytes',ctypes.c_uint32)]
class Params(ctypes.Structure):
 _fields_=[('norm',Norm),('ws',WS),('rank',ctypes.c_uint32),('num_counters',ctypes.c_uint32)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=P/a.out;out.mkdir(exist_ok=False)
 started=time.time();report=dict(status='STARTED',scope='SM86 singleGPU virtualpeer correctness only',checks=[],sources={n:sha(P/n) for n in ['attention_ar_norm.cuh','virtual_attention_actual.cu','probe_virtual_attention_actual.py']})
 def save(): (out/'results.json').write_text(json.dumps(report,indent=2))
 save()
 site=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages';cuda=site/'nvidia/cu13'
 inc=ROOT/'sglang/python/sglang/kernels/jit/include';muse=ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark'
 cmd=[str(cuda/'bin/nvcc'),'--cubin','-arch=sm_86','-std=c++20','-DSGL_CUDA_ARCH=860','-O3','--expt-relaxed-constexpr','--ptxas-options=-v','-I'+str(P),'-I'+str(inc),'-I'+str(muse),'-I'+str(site/'tvm_ffi/include'),'-I'+str(cuda/'include/cccl'),str(P/'virtual_attention_actual.cu'),'-o',str(out/'kernel.cubin')]
 (out/'command.json').write_text(json.dumps(cmd,indent=2));r=subprocess.run(cmd,capture_output=True,text=True,timeout=75)
 (out/'compiler.stdout.log').write_text(r.stdout);(out/'compiler.stderr.log').write_text(r.stderr)
 report['compile_exit']=r.returncode;save();assert r.returncode==0,r.stderr[-4000:]
 import torch,cuda.bindings.driver as d
 assert torch.cuda.get_device_capability()==(8,6)
 torch.set_num_threads(1)
 free,total=torch.cuda.mem_get_info();report['initial_free_bytes']=free
 assert free>=512*1024**2+96*1024**2,'512MiB reserve plus96MiB fixture budget'
 rc,module=d.cuModuleLoad(str(out/'kernel.cubin').encode());assert rc==0
 functions={}
 for n in ['abi_sizes','rank_order_reference','virtual_fused','virtual_reference_norm','virtual_prefilled_consumer','prefill_planes']:
  rc,f=d.cuModuleGetFunction(module,n.encode());assert rc==0;functions[n]=f
 report['kernel_sha256']=sha(out/'kernel.cubin')
 props=torch.cuda.get_device_properties(0);assert props.multi_processor_count>=8
 def launch(n,args,types,grid,threads=256,coop=False):
  f=functions[n];stream=torch.cuda.current_stream().cuda_stream
  if coop:
   rc,blocks=d.cuOccupancyMaxActiveBlocksPerMultiprocessor(f,threads,0);assert rc==0 and blocks>=1
   assert grid[0]*grid[1]<=blocks*props.multi_processor_count
   rc,=d.cuLaunchCooperativeKernel(f,grid[0],grid[1],1,threads,1,1,0,stream,(args,types));assert rc==0
  else:
   c=d.CUlaunchConfig();c.gridDimX,c.gridDimY,c.gridDimZ=grid[0],grid[1],1;c.blockDimX,c.blockDimY,c.blockDimZ=threads,1,1;c.sharedMemBytes=0;c.hStream=stream
   rc,=d.cuLaunchKernelEx(c,f,(args,types),0);assert rc==0
 ptr=ctypes.c_void_p;i32=ctypes.c_int32
 sizes=torch.zeros(5,dtype=torch.int64,device='cuda');launch('abi_sizes',(sizes.data_ptr(),),(ptr,),(1,1));actual=sizes.cpu().tolist();expected=[ctypes.sizeof(Norm),ctypes.sizeof(WS),ctypes.sizeof(Params),Params.rank.offset,Norm.pre_norm_bf16.offset];assert actual==expected,(actual,expected);report['ABI']=actual
 torch.cuda.reset_peak_memory_stats()
 for m in [1,2,8]:
  shape=(8,m,8192);partials=torch.empty(shape,dtype=torch.bfloat16,device='cuda');stream=torch.empty(shape,dtype=torch.float32,device='cuda');reduced=torch.empty_like(partials)
  outputs=[[torch.empty_like(partials),torch.empty_like(stream),torch.empty_like(stream)] for _ in range(2)]
  weight=torch.empty(8192,dtype=torch.bfloat16,device='cuda');pre=torch.empty_like(weight);alpha=torch.empty(8192,dtype=torch.float32,device='cuda');beta=torch.empty_like(alpha)
  slot=m*8192*2;plane=torch.zeros(8*16*slot,dtype=torch.uint8,device='cuda');counters=torch.zeros((8,132),dtype=torch.uint32,device='cuda')
  norms=[];pars=[]
  for rank in range(8):
   def norm(which,branch):
    return Norm(branch.data_ptr(),stream[rank].data_ptr(),outputs[which][0][rank].data_ptr(),outputs[which][1][rank].data_ptr(),outputs[which][2][rank].data_ptr(),0,pre.data_ptr(),alpha.data_ptr(),beta.data_ptr(),1e-8,1e-5,m,True,True,False)
   norms.append(norm(0,reduced[rank]));ws=WS((ctypes.c_void_p*8)(*[plane.data_ptr()+dst*16*slot for dst in range(8)]),counters[rank].data_ptr(),0,slot);pars.append(Params(norm(1,partials[rank]),ws,rank,132))
  def gpu_struct(values,typ):return torch.tensor(list(bytes((typ*8)(*values))),dtype=torch.uint8,device='cuda')
  norm_gpu=gpu_struct(norms,Norm);param_gpu=gpu_struct(pars,Params)
  assert len({t.data_ptr() for group in outputs for t in group})==6
  for generation in range(5):
   torch.manual_seed(9000+m*10+generation)
   partials.copy_(torch.randn(shape,device='cuda').to(torch.bfloat16));stream.normal_();weight.copy_(torch.randn_like(weight)*.1+1);pre.copy_(torch.randn_like(pre)*.1+1);gate=torch.randn_like(alpha)/.3;beta.copy_(torch.sigmoid(gate));alpha.copy_(torch.sqrt(torch.clamp(torch.sigmoid(-gate)*(1+beta),min=1e-3)))
   if generation==1:partials.zero_();partials[1::2].neg_()
   if generation==2:
    partials.zero_();partials[0].fill_(2**24);partials[1].fill_(1);partials[2].fill_(-2**24);partials[3].fill_(1)
   for group in outputs:
    for t in group:t.fill_(float('nan'))
   launch('rank_order_reference',(partials.data_ptr(),reduced.data_ptr(),m),(ptr,ptr,i32),((m*1024+255)//256,1))
   launch('virtual_reference_norm',(norm_gpu.data_ptr(),),(ptr,),(m,8))
   if m==1:
    launch('virtual_fused',(param_gpu.data_ptr(),),(ptr,),(m,8),coop=True)
    mode='cooperative_full_push_poll'
   else:
    plane.zero_();counters.zero_();launch('prefill_planes',(partials.data_ptr(),plane.data_ptr(),m,slot),(ptr,ptr,i32,i32),((8*m*1024+255)//256,1));launch('virtual_prefilled_consumer',(param_gpu.data_ptr(),),(ptr,),(m,8));mode='prepopulated_plane_consumer_only'
   torch.cuda.synchronize()
   for label,x,y in zip(['bf16_output','fp32_stream','fp32_router'],outputs[0],outputs[1]):
    assert torch.isfinite(x).all().item() and torch.isfinite(y).all().item()
    bits=torch.int16 if x.dtype==torch.bfloat16 else torch.int32
    same=torch.equal(x.view(bits),y.view(bits));report['checks'].append(dict(M=m,generation=generation,mode=mode,output=label,finite=True,bitwise=same));save();assert same,(m,generation,label,(x-y).abs().max().item())
   assert not plane.any().item(),'all used plane words reset'
   expected_phase=(generation+1)%2 if m==1 else 1
   assert torch.equal(counters,torch.full_like(counters,expected_phase)),'all132 counters coherent'
  del partials,stream,reduced,outputs,weight,pre,alpha,beta,plane,counters,norm_gpu,param_gpu
 report.update(status='PASS_LOCAL_VIRTUAL_PEER_ONLY',peak_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-started,no_performance_claim=True)
 save();print(json.dumps(report,indent=2))
if __name__=='__main__':main()

