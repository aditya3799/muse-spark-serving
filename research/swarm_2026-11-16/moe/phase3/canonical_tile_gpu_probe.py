"""Bounded exact-byte GPU diagnostic. No GEMM and no timing claim."""
import argparse, ctypes, hashlib, importlib.metadata as metadata, json, subprocess, sys, time
from pathlib import Path
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'preparing','scope':'Integer byte-layout diagnostic only; not GEMM or serving performance','checks':[],'source_hashes':{n:sha(P/n) for n in ('canonical_tile_adapter.cuh','canonical_tile_adapter.py','audit_canonical_tile.py','canonical_tile_gpu_probe.py')}}
def save():
 t=a.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.out/'results.json')
save()
# Compile exact frozen header once, with all four combinations of each architecture layout.
unit='#include <cuda_runtime.h>\n#include "'+str(P/'canonical_tile_adapter.cuh')+'"\n'
for gu in (0,1):
 for mul in (0,1):
  for tr in (0,1):
   unit+='extern "C" __global__ void tile_%d%d%d(const uint32_t* c,const uint8_t* s,uint8_t* o,uint32_t N,uint32_t u,uint32_t k) {canonical_tile::diagnostic_stage_body<%s,%s,%s>(c,s,o,N,u,k); }\n'%(gu,mul,tr,str(bool(gu)).lower(),str(bool(mul)).lower(),str(bool(tr)).lower())
(a.out/'kernel.cu').write_text(unit)
dist=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).endswith('/bin/nvcc'))
# RTX3050 local diagnostic only; SM90 compile is separate future evidence.
command=[str(nvcc),'--cubin','--gpu-architecture=sm_86','-std=c++17','-O3','-Xptxas=-v',str(a.out/'kernel.cu'),'-o',str(a.out/'kernel.sm86.cubin')]
report.update(status='compiling',compiler_command=command,compile_bound_seconds=120,runtime_bound_seconds=120);save()
start=time.monotonic()
r=subprocess.run(command,capture_output=True,text=True,timeout=120)
(a.out/'compiler_stdout.log').write_text(r.stdout);(a.out/'compiler_stderr.log').write_text(r.stderr)
report.update(compiler_exit=r.returncode,compile_seconds=time.monotonic()-start);save()
if r.returncode: report['status']='compile_failed';save();raise SystemExit(r.returncode)
report['cubin_sha256']=sha(a.out/'kernel.sm86.cubin')
# Runtime alarm is separate from the compiler wall-clock bound.
import signal
signal.alarm(120)
import torch
from cuda.bindings import driver as cbd
import audit_canonical_tile as ref
ref.DONOR=P.parents[3]/"muse_moe_w4a16/submission_v139_wgs_kperm3.py"
assert sha(ref.DONOR)=="a24b8374aad49fbaa5e0cdeffb7f1fa64960d8321f343cf8b88bd97dfdd20c09", "Pinned donor hash differs"
(a.out/"donor_source_snapshot.py").write_bytes(ref.DONOR.read_bytes())
report["donor_sha256"]=sha(ref.DONOR);save()
from canonical_tile_adapter import translate_stage
import ast
ns={'DQ7':True};tree=ast.parse(ref.DONOR.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_stored_pos','_stored_pos_mul')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ref.DONOR),'exec'),ns)
torch.cuda.init();context_anchor=torch.empty(1,device='cuda');report.update(status='running',gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),torch=torch.__version__);save()
assert torch.cuda.get_device_capability()==(8,6),'Frozen compile is SM86 only'
err,mod=cbd.cuModuleLoad(str(a.out/'kernel.sm86.cubin').encode());assert err==cbd.CUresult.CUDA_SUCCESS,err
functions={}
for gu in (0,1):
 for mul in (0,1):
  for tr in (0,1):
   err,f=cbd.cuModuleGetFunction(mod,('tile_%d%d%d'%(gu,mul,tr)).encode());assert err==cbd.CUresult.CUDA_SUCCESS;functions[gu,mul,tr]=f

def launch(f,c,s,o,N,u,k):
 args=(c.data_ptr(),s.data_ptr(),o.data_ptr()+128,N,u,k)
 types=(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32)
 result=cbd.cuLaunchKernel(f,1,1,1,128,1,1,0,torch.cuda.current_stream().cuda_stream,(args,types),0)
 assert result[0]==cbd.CUresult.CUDA_SUCCESS,result

def check(name,out,expected,context):
 actual=bytes(out[128:-128].cpu().tolist()); assert len(actual)==len(expected)==36864; mismatches=sum(x!=y for x,y in zip(actual,expected))
 guards=bool((out[:128]==165).all().item()) and bool((out[-128:]==165).all().item())
 row=dict(name=name,context=context,mismatches=mismatches,guards_intact=guards,expected_sha256=hashlib.sha256(expected).hexdigest(),utc_seconds=time.time())
 report['checks'].append(row)
 if mismatches or not guards:report['status']='byte_gate_failed'
 save();assert mismatches==0 and guards,row

for gu in (0,1):
 N=8192 if gu else 4096
 for u,k in ((0,0),(N//256-1,15)):
  rows=[(logical//16*8+logical%8+(4096 if logical%16>=8 else 0)) if gu else logical for logical in range(u*256,(u+1)*256)]
  for tr in (0,1):
   codes=bytearray(N*4096//2);scales=bytearray(N*256)
   for n0 in sorted({r//64*64 for r in rows}):
    for k0 in range(k*256,(k+1)*256,64):ref.pack_source_block(codes,n0,k0,N,bool(tr))
   for kg in range(k*16,(k+1)*16):
    for n0 in sorted({r//64*64 for r in rows}):
     perm=[x+i for i in range(8) for x in range(0,64,8)]
     for oj,sj in enumerate(perm):scales[kg*N+n0+oj]=ref.raw_scale(n0+sj,kg)
   dc=torch.frombuffer(codes,dtype=torch.uint8).clone().cuda();ds=torch.frombuffer(scales,dtype=torch.uint8).clone().cuda()
   for mul in (0,1):
    pos=ns['_stored_pos_mul' if mul else '_stored_pos']();expected=ref.oracle_stage(u,k,bool(gu),bool(mul),pos)
    out=torch.full((36864+256,),165,dtype=torch.uint8,device='cuda')
    f=functions[gu,mul,tr];context=dict(gate_up=gu,mul=mul,transpose=tr,unit=u,stage=k)
    launch(f,dc,ds,out,N,u,k);check('eager',out,expected,context)
    st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):launch(f,dc,ds,out,N,u,k)
    torch.cuda.current_stream().wait_stream(st);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):launch(f,dc,ds,out,N,u,k)
    dc.bitwise_not_();ds.bitwise_not_();out.fill_(165)
    origc,origs=ref.raw_code,ref.raw_scale
    ref.raw_code=lambda n,kk:origc(n,kk)^15;ref.raw_scale=lambda n,g:origs(n,g)^255
    changed=ref.oracle_stage(u,k,bool(gu),bool(mul),pos)
    ref.raw_code,ref.raw_scale=origc,origs
    graph.replay();check('changed_bytes_graph',out,changed,context)
    dc.bitwise_not_();ds.bitwise_not_()
    del graph,out
   del dc,ds
signal.alarm(0)
assert all(sha(P/n)==h for n,h in report['source_hashes'].items()),'Source changed during run'
report.update(status='passed',source_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'compile_seconds':report['compile_seconds']}),flush=True)
