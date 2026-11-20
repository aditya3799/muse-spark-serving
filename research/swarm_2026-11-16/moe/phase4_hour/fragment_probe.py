import argparse,ctypes,hashlib,importlib.metadata as md,json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
inc=P.parent/'humming_vendor/humming/include'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'preparing','checks':[],'scope':'Canonical BF16 fragment decode only. No GEMM or performance claim.','sources':{str(f):sha(f) for f in [P/'canonical_fragment.cuh',Path(__file__),*sorted(inc.rglob('*.cuh'))]}}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(r,indent=2)+'\n');t.replace(a.out/'results.json')
src='#include "'+str(P/'canonical_fragment.cuh')+'"\n'
for mode in range(3):src+='extern "C" __global__ void decode%d(const uint32_t*i,uint32_t*a,uint32_t*e,uint32_t n){fragment::run<%d>(i,a,e,n);}\n'%(mode,mode)
(a.out/'kernel.cu').write_text(src)
d=md.distribution('nvidia-cuda-nvcc');nvcc=next(Path(d.locate_file(f)).resolve() for f in d.files if str(f).endswith('/bin/nvcc'))
cmd=[str(nvcc),'--cubin','--gpu-architecture=sm_86','-std=c++17','--expt-relaxed-constexpr','--use_fast_math','-O3','-I'+str(inc),'-Xptxas=-v',str(a.out/'kernel.cu'),'-o',str(a.out/'kernel.cubin')]
r.update(status='compiling',command=cmd);save();start=time.monotonic();c=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
(a.out/'compile.log').write_text(c.stdout+c.stderr);r.update(compile_exit=c.returncode,compile_seconds=time.monotonic()-start);save()
if c.returncode:raise SystemExit(c.returncode)
import torch
from cuda.bindings import driver as cbd
anchor=torch.empty(1,device='cuda');e,mod=cbd.cuModuleLoad(str(a.out/'kernel.cubin').encode());assert e==cbd.CUresult.CUDA_SUCCESS
inputs=[]
for code in range(16):
 for scale in range(256):
  q=[sum(((code+j+delta)&15)<<(4*j) for j in range(8)) for delta in (0,7)]
  inputs.append([q[0],q[1],scale,255-scale])
x=torch.tensor(inputs,dtype=torch.int64).to(torch.uint32).cuda();count=len(inputs)
actual=torch.empty((count,4),device='cuda',dtype=torch.uint32);expected=torch.empty_like(actual)
mag=torch.tensor([0,.5,1,1.5,2,3,4,6],device='cuda',dtype=torch.float32)
def record(name,a,b,**info):
 mismatches=int((a!=b).sum().item());row=dict(name=name,mismatches=mismatches,elements=a.numel(),info=info,utc_seconds=time.time());r['checks'].append(row)
 if mismatches:r['status']='failed';row['first_mismatch']=int((a!=b).flatten().nonzero()[0])
 save();assert mismatches==0,row
for mode in range(3):
 e,f=cbd.cuModuleGetFunction(mod,('decode%d'%mode).encode());assert e==cbd.CUresult.CUDA_SUCCESS
 def launch():
  v=(x.data_ptr(),actual.data_ptr(),expected.data_ptr(),count);t=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32)
  e=cbd.cuLaunchKernel(f,(count+127)//128,1,1,128,1,1,0,torch.cuda.current_stream().cuda_stream,(v,t),0)[0];assert e==cbd.CUresult.CUDA_SUCCESS
 launch();record('pinned_fragment_eager',actual,expected,mode=mode)
 # Independent mathematical reference for all finite E4M3 scale encodings.
 xin=x.to(torch.int64);sf=xin[:,2:4].to(torch.uint8).contiguous().view(torch.float8_e4m3fn).float()
 gold=[];finite_masks=[]
 for reg in range(4):
  q=xin[:,0 if mode==0 or reg%2==0 else 1];shift=reg*4 if mode==0 else (mode-1)*4+(reg//2)*8
  for high in range(2):
   c=(q>>(shift+16*high))&15;v=mag[c&7]*torch.where(c<8,1.,-1.);gold.append((v*.25*sf[:,reg%2]).to(torch.bfloat16));finite_masks.append(torch.isfinite(sf[:,reg%2]))
 golden=torch.stack(gold,dim=1).contiguous();finite=torch.stack(finite_masks,dim=1)
 record('finite_math_reference',actual.view(torch.int16)[finite],golden.view(torch.int16)[finite],mode=mode,scale_encodings=254)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):launch()
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):launch()
 x.copy_(x.flip(0));g.replay();record('changed_input_graph',actual,expected,mode=mode);x.copy_(x.flip(0))
 del g
r.update(status='passed',gpu=torch.cuda.get_device_name(),epilogue_compensation=4,internal_weight_factor=.25,source_unchanged=all(sha(Path(f))==h for f,h in r['sources'].items()));save();assert r['source_unchanged'];print(json.dumps({'status':r['status'],'checks':len(r['checks'])}))
