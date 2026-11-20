import ctypes as C,json,hashlib
from pathlib import Path
import torch
from validate_alignment import ROOT,preamble,compile_kernels,launch,p,I,route_set,align,NAMES
import validate_alignment as a
source=ROOT/'sglang/python/sglang/kernels/jit/csrc/moe/align_single_token.cuh'
s=source.read_text(encoding='utf-8')
body=s[s.index('struct AlignSingleTokenParams'):s.index('template <bool kUsePDL, bool kWriteInactivePrefixCount = false>\nstruct AlignSingleTokenKernel')]
# Only the unused PDL=false helpers and host-framework includes are stubbed.
helpers='namespace device { template<bool B> __device__ void PDLWaitPrimary() {} template<bool B> __device__ void PDLTriggerSecondary() {} }\n'
code='typedef unsigned int uint32_t;\n'+preamble+'\n#define INT32_MAX 2147483647\nnamespace sglang {\n'+helpers+body+'\n}\n'
name='&sglang::align_single_token_kernel<false,false>'
class Params(C.Structure):
 _fields_=[('ids',C.c_void_p),('sorted',C.c_void_p),('experts',C.c_void_p),('count',C.c_void_p),('topk',C.c_uint32),('block',C.c_uint32)]
fn=compile_kernels(code,[name])[name]
a.FUNCS=compile_kernels(a.preamble+a.body,NAMES)
def single(ids,block):
 k=ids.numel(); out=torch.empty(k*block,device='cuda',dtype=torch.int32); experts=torch.empty(k,device='cuda',dtype=torch.int32); count=torch.empty(1,device='cuda',dtype=torch.int32)
 params=Params(ids.data_ptr(),out.data_ptr(),experts.data_ptr(),count.data_ptr(),k,block)
 launch(fn,1,32,0,[params]); return out,experts,count
cases=0
torch.manual_seed(27)
for local in [1,4,32,64]:
 for topk in [1,2,8,16,32]:
  for block in [8,16,32,64]:
   for mode in ['mixed','all_local','all_remote']:
    raw=torch.randperm(256)[:topk].to(torch.int32)
    offset=256-local
    if mode=='all_local': raw=torch.arange(topk,dtype=torch.int32)%local+offset
    if mode=='all_remote': raw.fill_(0)
    ids=torch.where((raw>=offset)&(raw<offset+local),raw-offset,-1).view(1,-1).cuda()
    expected=sorted((int(e),j) for j,e in enumerate(ids.flatten().cpu().tolist()) if e>=0)
    assert route_set(single(ids,block),ids,block)==expected
    assert route_set(align(ids,local,block),ids,block)==expected
    cases+=1
# Verify captured outputs track changed input IDs without stale scratch state.
ids=torch.tensor([[-1,0,-1,31,7,-1,3,-1]],device='cuda',dtype=torch.int32)
stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream): single(ids,8)
torch.cuda.current_stream().wait_stream(stream)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g): table=single(ids,8)
for step in range(8):
 ids.copy_(torch.tensor([[step%32,-1,31,-1,7,0,-1,3]],device='cuda',dtype=torch.int32))
 g.replay(); expected=sorted((int(e),j) for j,e in enumerate(ids.flatten().cpu().tolist()) if e>=0)
 assert route_set(table,ids,8)==expected
result={'cases':cases,'graph_replays':8,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'existing single-token CUDA body, including slot tie-break, compiled with NVRTC; PDL=false on local Ampere; mixed and repeated negative IDs'}
(ROOT/'validation/single_token_iteration3.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

