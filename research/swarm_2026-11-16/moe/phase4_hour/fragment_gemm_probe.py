import argparse,ctypes,hashlib,importlib.metadata as md,json,subprocess,sys,time,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent/'phase3'))
import audit_canonical_tile as ref
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
inc=P.parent/'humming_vendor/humming/include';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'preparing','scope':'Controlled SM86 MMA integration ofcanonicalHopperlayout, sameFP32sequence inbothvariants. NotproductionHummingGEMM, notH200timing.','checks':[],'timings':[],'sources':{str(f):sha(f) for f in [P/'canonical_fragment.cuh',P/'fragment_gemm.cuh',P.parent/'phase3/canonical_tile_adapter.cuh',Path(__file__),*sorted(inc.rglob('*.cuh'))]}}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.out/'results.json')
src='#include "'+str(P/'fragment_gemm.cuh')+'"\n'
for direct in (0,1):src+='extern "C" __global__ void gemm%d(const uint32_t*c,const uint8_t*s,const nv_bfloat16*x,float*a,nv_bfloat16*r,nv_bfloat16*f,uint32_t M,uint32_t N,uint32_t K,uint16_t gs){fragment_gemm::run<%s>(c,s,x,a,r,f,M,N,K,*reinterpret_cast<nv_bfloat16*>(&gs));}\n'%(direct,str(bool(direct)).lower())
(a.out/'kernel.cu').write_text(src)
d=md.distribution('nvidia-cuda-nvcc');nvcc=next(Path(d.locate_file(f)).resolve() for f in d.files if str(f).endswith('/bin/nvcc'))
cmd=[str(nvcc),'--cubin','--gpu-architecture=sm_86','-std=c++17','--expt-relaxed-constexpr','--use_fast_math','-O3','-I'+str(inc),'-Xptxas=-v',str(a.out/'kernel.cu'),'-o',str(a.out/'kernel.cubin')]
r.update(status='compiling',command=cmd);save();start=time.monotonic();c=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(a.out/'compile.log').write_text(c.stdout+c.stderr);r.update(compile_exit=c.returncode,compile_seconds=time.monotonic()-start);save()
if c.returncode:raise SystemExit(c.returncode)
import torch
from cuda.bindings import driver as cbd
anchor=torch.empty(1,device='cuda');e,mod=cbd.cuModuleLoad(str(a.out/'kernel.cubin').encode());assert e==cbd.CUresult.CUDA_SUCCESS
fs=[]
for i in (0,1):e,f=cbd.cuModuleGetFunction(mod,('gemm%d'%i).encode());assert e==cbd.CUresult.CUDA_SUCCESS;fs.append(f)
N=64;K=4096;codes=bytearray(N*K//2);scales=bytearray(N*K//16)
for n0 in range(0,N,64):
 for k0 in range(0,K,64):ref.pack_source_block(codes,n0,k0,N,True)
# Low normal and subnormal FP8scales, positive/negative/zero; no NaNs in GEMM inputs.
from canonical_tile_adapter import humming_scale_address
for n in range(N):
 for kg in range(K//16):scales[humming_scale_address(n,kg,N)]=(0,1,8,16,24,32,136,144)[(n+kg)%8]
dc=torch.frombuffer(codes,dtype=torch.uint8).clone().cuda();ds=torch.frombuffer(scales,dtype=torch.uint8).clone().cuda()

def record(name,x,y,**info):
 mismatches=int((x.view(torch.uint8)!=y.view(torch.uint8)).sum().item());row=dict(name=name,mismatches=mismatches,bytes=x.numel()*x.element_size(),info=info,utc_seconds=time.time());r['checks'].append(row)
 if mismatches:r['status']='failed'
 save();assert mismatches==0,row

for M in (1,8,33):
 torch.manual_seed(723+M);x=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)*.25
 outs=[(torch.empty((M,N),device='cuda',dtype=torch.float32),torch.empty((M,N),device='cuda',dtype=torch.bfloat16),torch.empty((M,N),device='cuda',dtype=torch.bfloat16)) for _ in fs]
 gs=int(torch.tensor(1.5,dtype=torch.bfloat16).view(torch.uint16))
 def launch(i):
  accum,raw,final=outs[i];v=(dc.data_ptr(),ds.data_ptr(),x.data_ptr(),accum.data_ptr(),raw.data_ptr(),final.data_ptr(),M,N,K,gs)
  t=(ctypes.c_void_p,)*6+(ctypes.c_uint32,)*3+(ctypes.c_uint16,)
  e=cbd.cuLaunchKernel(fs[i],N//16,(M+7)//8,1,32,1,1,0,torch.cuda.current_stream().cuda_stream,(v,t),0)[0];assert e==cbd.CUresult.CUDA_SUCCESS
 for i in (0,1):launch(i)
 for j,name in enumerate(('fp32_accumulator','raw_bf16','scaled_bf16')):record(name,outs[0][j],outs[1][j],M=M)
 # Independent matrix-orientation oracle; fixed tolerance is diagnostic only.
 vals=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
 nib=torch.tensor([[ref.raw_code(n,k) for k in range(K)] for n in range(N)],device='cuda')
 sf=torch.tensor([[(0,1,8,16,24,32,136,144)[(n+kg)%8] for kg in range(K//16)] for n in range(N)],device='cuda',dtype=torch.uint8).view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=1)
 torch.backends.cuda.matmul.allow_tf32=False;gold=x.float()@(vals[nib]*sf).T
 err=float((gold-outs[1][0]).abs().max().item());row={'name':'independent_fp32_orientation','M':M,'max_abs':err,'fixed_limit':.0001,'passed':err<=.0001};r['checks'].append(row);save();assert row['passed'],row
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for i in (0,1):launch(i)
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graphs=[]
 for i in (0,1):
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   for _ in range(8):launch(i)
  graphs.append(g)
 x.neg_()
 for g in graphs:g.replay()
 for j,name in enumerate(('graph_fp32','graph_raw','graph_scaled')):record(name,outs[0][j],outs[1][j],M=M)
 # Balanced paired graph timings. Decoder loop comparison only.
 samples=[[],[]]
 for trial in range(8):
  for i in ((0,1) if trial%2==0 else (1,0)):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();graphs[i].replay();end.record();end.synchronize();samples[i].append(start.elapsed_time(end)*1000/8)
 r['timings'].append({'M':M,'reference_us':statistics.median(samples[0]),'direct_us':statistics.median(samples[1]),'ratio':statistics.median(samples[0])/statistics.median(samples[1]),'scope':'localcontrolledloop; noA/AorH200servingclaim'});save()
 del graphs
r.update(status='passed',gpu=torch.cuda.get_device_name(),source_unchanged=all(sha(Path(f))==h for f,h in r['sources'].items()));save();assert r['source_unchanged'];print(json.dumps({'status':r['status'],'checks':len(r['checks']),'timings':r['timings']}))
