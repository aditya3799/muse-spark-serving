"""Validate the actual router class with a local BF16->FP32 cuBLAS adapter.
PyTorch 2.6 lacks mm(out_dtype=...); the judge's newer PyTorch provides it.
This adapter exists only in the local test, never in the submitted patch.
"""
import ast, ctypes, json, statistics, sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'sglang/python/sglang/srt/models/muse_spark.py'
lib=ctypes.WinDLL(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\cublas64_12.dll')
handle=ctypes.c_void_p()
def check(status):
 if status: raise RuntimeError('cuBLAS status '+str(status))
check(lib.cublasCreate_v2(ctypes.byref(handle)))
lib.cublasSetStream_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
lib.cublasGemmEx.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
def mm(x,w,*,out_dtype):
 assert out_dtype==torch.float32 and x.dtype==w.dtype==torch.bfloat16
 assert x.is_contiguous() and w.is_contiguous()
 m,k=x.shape; kk,n=w.shape; assert kk==k
 out=torch.empty((m,n),device=x.device,dtype=torch.float32)
 alpha,beta=ctypes.c_float(1),ctypes.c_float(0)
 check(lib.cublasSetStream_v2(handle,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
 check(lib.cublasGemmEx(handle,0,0,n,m,k,ctypes.byref(alpha),ctypes.c_void_p(w.data_ptr()),14,n,ctypes.c_void_p(x.data_ptr()),14,k,ctypes.byref(beta),ctypes.c_void_p(out.data_ptr()),0,n,68,-1))
 return out
class TorchAdapter:
 def __getattr__(self,k): return getattr(torch,k)
 mm=staticmethod(mm)
def router_class(text):
 tree=ast.parse(text)
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xRouter')
 ns={'torch':TorchAdapter(),'nn':nn}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
 return ns['MuseSpark1xRouter']
def stock(router,x):
 w=router.weight.detach(); hi=w.to(torch.bfloat16); lo=(w-hi.float()).to(torch.bfloat16)
 return mm(x,hi.t().contiguous(),out_dtype=torch.float32)+mm(x,lo.t().contiguous(),out_dtype=torch.float32)
def relative(a,b): return ((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30)).item()
def run():
 torch.manual_seed(179); torch.backends.cuda.matmul.allow_tf32=False
 cls=router_class(SOURCE.read_text(encoding='utf-8'))
 records=[]
 for n,k in [(256,8192),(37,257)]:
  r=cls(k,n,torch.float32).cuda()
  with torch.no_grad(): r.weight.normal_()
  for m in [1,7,33,64,128,256,513]:
   x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
   # Include one-hot probes and near-zero inputs.
   x[0].zero_(); x[0,k//2]=1
   if m>1: x[1]*=2**-16
   ref=stock(r,x); out=r.forward_split2(x)
   err=relative(out,ref)
   oracle=x.double()@r.weight.double().T
   original_error=relative(ref,oracle); candidate_error=relative(out,oracle)
   assert err<1e-4,(m,n,k,err)
   assert candidate_error < 1e-4 and candidate_error <= original_error + 1e-5,(original_error,candidate_error)
   assert out.dtype==torch.float32 and out.shape==(m,n) and torch.isfinite(out).all()
   records.append({'m':m,'n':n,'k':k,'vs_stock_rel_l2':err,'vs_fp64_rel_l2':candidate_error})
 # Capture/replay with changing inputs and concurrent graph instances.
 r=cls(8192,256,torch.float32).cuda()
 with torch.no_grad(): r.weight.normal_()
 inputs=[torch.randn(128,8192,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
 graphs=[]; outputs=[]
 for x in inputs:
  stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(3): r.forward_split2(x)
  torch.cuda.current_stream().wait_stream(stream)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g): y=r.forward_split2(x)
  graphs.append(g); outputs.append(y)
 for step in range(4):
  for x,g,y in zip(inputs,graphs,outputs):
   x.normal_(); g.replay(); assert relative(y,stock(r,x))<1e-4
 torch.cuda.synchronize()
 result={'device':torch.cuda.get_device_name(),'torch':torch.__version__,'method':'actual extracted router class; cuBLAS BF16 operands/FP32 accumulators; no judge run','cases':records,'graph_replays':8,'max_vs_stock_rel_l2':max(x['vs_stock_rel_l2'] for x in records)}
 target=ROOT/'validation'; target.mkdir(exist_ok=True)
 (target/'router_iteration1.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
if __name__=='__main__': run()


