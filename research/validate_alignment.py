"""Run SGLang's existing alignment CUDA bodies locally via NVRTC.
No SGLang installation or CUDA source in the submission is changed by this test.
"""
import ctypes as C, json, hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
BIN=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin')
nv=C.WinDLL(str(BIN/'nvrtc64_120_0.dll')); cu=C.WinDLL('nvcuda.dll')
ptr=C.c_void_p; integer=C.c_int
nv.nvrtcCreateProgram.argtypes=[C.POINTER(ptr),C.c_char_p,C.c_char_p,integer,ptr,ptr]
nv.nvrtcAddNameExpression.argtypes=[ptr,C.c_char_p]
nv.nvrtcCompileProgram.argtypes=[ptr,integer,C.POINTER(C.c_char_p)]
nv.nvrtcGetProgramLogSize.argtypes=[ptr,C.POINTER(C.c_size_t)]
nv.nvrtcGetProgramLog.argtypes=[ptr,ptr]
nv.nvrtcGetPTXSize.argtypes=[ptr,C.POINTER(C.c_size_t)]
nv.nvrtcGetPTX.argtypes=[ptr,ptr]
nv.nvrtcGetLoweredName.argtypes=[ptr,C.c_char_p,C.POINTER(C.c_char_p)]
cu.cuModuleLoadData.argtypes=[C.POINTER(ptr),ptr]
cu.cuModuleGetFunction.argtypes=[C.POINTER(ptr),ptr,C.c_char_p]
cu.cuLaunchKernel.argtypes=[ptr,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,ptr,C.POINTER(ptr),ptr]
def check(rc):
 if rc: raise RuntimeError('CUDA/NVRTC status '+str(rc))
def compile_kernels(source,names):
 torch.empty(1,device='cuda')
 program=ptr(); check(nv.nvrtcCreateProgram(C.byref(program),source.encode(),b'align_validation.cu',0,None,None))
 for name in names: check(nv.nvrtcAddNameExpression(program,name.encode()))
 major,minor=torch.cuda.get_device_capability()
 options=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++17']
 rc=nv.nvrtcCompileProgram(program,len(options),(C.c_char_p*len(options))(*options))
 log_size=C.c_size_t(); check(nv.nvrtcGetProgramLogSize(program,C.byref(log_size)))
 log=C.create_string_buffer(log_size.value); check(nv.nvrtcGetProgramLog(program,log))
 if rc: raise RuntimeError(log.value.decode())
 size=C.c_size_t(); check(nv.nvrtcGetPTXSize(program,C.byref(size))); ptx=C.create_string_buffer(size.value); check(nv.nvrtcGetPTX(program,ptx))
 module=ptr(); check(cu.cuModuleLoadData(C.byref(module),ptx)); funcs={}
 for name in names:
  lowered=C.c_char_p(); check(nv.nvrtcGetLoweredName(program,name.encode(),C.byref(lowered)))
  fn=ptr(); check(cu.cuModuleGetFunction(C.byref(fn),module,lowered.value)); funcs[name]=fn
 return funcs

def launch(fn,grid,threads,shared,args):
 addresses=(ptr*len(args))(*(C.cast(C.byref(a),ptr) for a in args))
 check(cu.cuLaunchKernel(fn,grid,1,1,threads,1,1,shared,ptr(torch.cuda.current_stream().cuda_stream),addresses,None))
def p(t): return ptr(t.data_ptr())
I=integer; Z=C.c_size_t; B=C.c_bool
AOT=ROOT/'sglang/python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu'
s=AOT.read_text(encoding='utf-8'); body=s[s.index('#define VEC_SIZE'):s.index('\nvoid moe_align_block_size(')]
preamble='typedef int int32_t; typedef long long int64_t;\n#define WARP_SIZE 32\n#define CEILDIV(x,y) (((x)+(y)-1)/(y))\n'
NAMES=['&count_and_sort_expert_tokens_kernel<int>','&moe_align_block_size_kernel<int>','&moe_align_block_size_small_batch_expert_kernel<int,256>']
FUNCS=None

def align(ids,experts,block):
 n=ids.numel(); buckets=experts+1
 capacity=n*block if n<buckets else n+buckets*(block-1)
 capacity=(capacity+3)//4*4
 sorted_ids=torch.full((capacity,),-777,device='cuda',dtype=torch.int32)
 expert_ids=torch.full(((capacity+block-1)//block,),-777,device='cuda',dtype=torch.int32)
 count=torch.empty(1,device='cuda',dtype=torch.int32)
 cumsum=torch.empty(buckets+1,device='cuda',dtype=torch.int32)
 if n<1024 and buckets<=64:
  nt=max(buckets,32); shared=((nt+1)*buckets+buckets+1)*4
  args=[p(ids),p(sorted_ids),p(expert_ids),p(count),I(buckets),I(block),Z(n),B(True),B(False),I(capacity)]
  launch(FUNCS[NAMES[2]],1,256+nt,shared,args)
 else:
  scan=1<<(buckets-1).bit_length(); shared=(buckets+buckets+1+scan+32)*4
  args=[p(ids),p(sorted_ids),p(expert_ids),p(count),I(buckets),I(block),Z(n),p(cumsum),B(True),B(False),I(scan),I(capacity)]
  launch(FUNCS[NAMES[1]],2,1024,shared,args)
  launch(FUNCS[NAMES[0]],(n+255)//256,256,0,[p(ids),p(sorted_ids),p(cumsum),Z(n),B(False)])
 return sorted_ids,expert_ids,count

def route_set(table,ids,block):
 sorted_ids,experts,count=table
 n=ids.numel(); used=int(count.item()); assert used%block==0 and used<=sorted_ids.numel()
 sorted_ids=sorted_ids[:used].cpu().tolist(); experts=experts[:used//block].cpu().tolist()
 result=[]
 for b,e in enumerate(experts):
  for route in sorted_ids[b*block:(b+1)*block]:
   assert route>=0
   if route<n:
    assert ids.view(-1)[route].item()==e
    if e>=0: result.append((e,route))
 return sorted(result)

def run():
 global FUNCS
 FUNCS=compile_kernels(preamble+body,NAMES)
 gen=torch.Generator().manual_seed(19); cases=0; samples=[]
 for local in [1,4,16,32,64]:
  for m in [1,2,7,16,32,127,128,256,513]:
   for block in [8,16,32,64]:
    raw=torch.randint(-1,256,(m,8),generator=gen,dtype=torch.int32)
    offset=(256//local-1)*local
    ids=torch.where((raw>=offset)&(raw<offset+local),raw-offset,-1).cuda()
    # Deliberately exercise active, inactive, repeated and boundary expert IDs.
    ids.view(-1)[0]=local-1
    if m==1 and block==64: ids.fill_(-1)
    expected=sorted((int(e),j) for j,e in enumerate(ids.flatten().cpu().tolist()) if e>=0)
    old=align(ids,256,block); new=align(ids,local,block)
    assert route_set(old,ids,block)==expected
    assert route_set(new,ids,block)==expected
    assert int(old[2].item())==int(new[2].item())
    cases+=1
    if local==32 and block==8: samples.append({'m':m,'old_capacity':old[0].numel(),'new_capacity':new[0].numel()})
 result={'device':torch.cuda.get_device_name(),'cases':cases,'method':'unmodified AOT CUDA kernel bodies compiled with NVRTC; original vs local bucket counts; active route identity checked','source_sha256':hashlib.sha256(AOT.read_bytes()).hexdigest(),'capacity_examples':samples}
 (ROOT/'validation/align_iteration2.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__': run()
