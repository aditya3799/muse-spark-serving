"""Runtime full-shape discovery and complete-column parity calibration for native BF16 linear.

Keep the original replicated layer for unknown shapes and exact fallback. The fast path
is empirical until a matching native reduction order is independently established.
"""
import ctypes as ct
from pathlib import Path
from functools import lru_cache

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def runtime_lib():
    lib = ct.CDLL(str(HERE/"runtime_lt.so"))
    ptr,num,size=ct.c_void_p,ct.c_int,ct.c_uint64
    for name,signature,returns in [
        ("projection_prepare",[num,num,num,num,size],ptr),("projection_count",[ptr],num),
        ("projection_lt_version",[],size),("projection_attr",[ptr,num,num],size),
        ("projection_call",[ptr,num,num,ptr,ptr,ptr,ptr,size,ptr],num),("projection_drop",[ptr],None)]:
        fn=getattr(lib,name)
        fn.argtypes,fn.restype=signature,returns
    return lib


@lru_cache(maxsize=2)
def workspace_accessor(blas):
    symbol={"cublas":"_ZN2at4cuda22getChosenWorkspaceSizeEv",
            "cublaslt":"_ZN2at4cuda26getCUDABlasLtWorkspaceSizeEv"}.get(blas)
    if symbol is None: return None
    native=ct.CDLL(str(Path(torch.__file__).resolve().parent/"lib/libtorch_cuda.so"))
    fn=getattr(native,symbol)
    fn.argtypes,fn.restype=[],ct.c_size_t
    return fn


def native_workspace():
    # Cache only the symbol. The active override is re-read on each call.
    try:
        blas=str(torch.backends.cuda.preferred_blas_library()).split(".")[-1].lower()
        fn=workspace_accessor(blas)
        return None if fn is None else (blas,int(fn()))
    except (OSError,AttributeError,RuntimeError):
        return None


def compatible(x,w,world):
    return (x.is_cuda and w.device==x.device and torch.cuda.current_device()==x.device.index
            and x.dtype==w.dtype==torch.bfloat16 and x.ndim==w.ndim==2
            and x.shape[0]>0 and w.shape[1]==x.shape[1] and world>1 and w.shape[0]%world==0
            and x.is_contiguous() and w.is_contiguous() and x.data_ptr()%256==0 and w.data_ptr()%256==0
            and (w.shape[0]//world)*w.shape[1]*w.element_size()%256==0)


class VendorPlan:
    def __init__(self,lib,handle,world,n,m,device):
        self.lib,self.handle,self.world=lib,handle,world
        self.n,self.m,self.device=n,m,device
        self.index=None
        self.workspace=None
        self.stream=None
        self.k=None

    def close(self):
        if self.handle:
            self.lib.projection_drop(self.handle)
            self.handle=None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def choices(self):
        return self.lib.projection_count(self.handle)

    def details(self,index):
        return dict(zip(("vendor_ordinal","local_workspace","full_workspace"),
                    [self.lib.projection_attr(self.handle,index,i) for i in range(3)]))

    def invoke(self,index,full,w,x,out,temp):
        return self.lib.projection_call(self.handle,index,int(full),w.data_ptr(),x.data_ptr(),out.data_ptr(),
            temp.data_ptr() if temp.numel() else None,temp.numel(),torch.cuda.current_stream().cuda_stream)

    def local(self,x,w,rank):
        if self.index is None or not compatible(x,w,self.world) or x.device!=self.device:
            return None
        if (x.shape[0],x.shape[1],w.shape[0])!=(self.m,self.k,self.n) or not 0<=rank<self.world:
            return None
        if torch.cuda.current_stream().cuda_stream!=self.stream: return None
        width=self.n//self.world
        out=torch.empty(self.m,width,device=x.device,dtype=x.dtype)
        status=self.invoke(self.index,False,w.narrow(0,rank*width,width),x,out,self.workspace)
        return None if status else out


@torch.no_grad()
def discover(examples,world):
    """Examples share full geometry and device, with genuinely different weights/inputs.

    No static IDs. Try the full-operation heuristic order, using this process's active
    backend/workspace. A retained plan must reproduce full native output and every shard.
    """
    examples=list(examples)
    if not examples: return None,{"reason":"no calibration examples"}
    x,w=examples[0]
    if not all(compatible(v,wt,world) and v.shape==x.shape and wt.shape==w.shape
               and v.device==x.device for v,wt in examples): return None,{"reason":"unsupported layout"}
    if torch.cuda.is_current_stream_capturing(): return None,{"reason":"new plan during capture"}
    actual=native_workspace()
    if actual is None: return None,{"reason":"unknown torch backend/workspace"}
    try: lib=runtime_lib()
    except OSError: return None,{"reason":"unavailable native plan library"}
    prop=torch.cuda.get_device_properties(x.device)
    backend,size=actual
    information={"blas":backend,"native_workspace":size,"cublasLt_version":lib.projection_lt_version(),
        "device":prop.name,"sm_count":prop.multi_processor_count,"capability":[prop.major,prop.minor],
        "torch":str(torch.__version__),"cuda_runtime":torch.version.cuda,"world":world,
        "shape":[x.shape[0],w.shape[0],w.shape[1]],"calibration_examples":len(examples),
        "reduced_bf16":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    m,k=x.shape
    n=w.shape[0]
    handle=lib.projection_prepare(m,n,k,world,size)
    if not handle: return None,{**information,"reason":"no exposed plan"}
    plan=VendorPlan(lib,handle,world,n,m,x.device)
    try:
        refs=[F.linear(v,wt) for v,wt in examples]
        full,local=torch.empty_like(refs[0]),torch.empty((m,n//world),device=x.device,dtype=x.dtype)
        information["eligible"]=plan.choices()
        tried=[]
        information["tried"]=tried
        for i in range(plan.choices()):
            info=plan.details(i)
            scratch=torch.empty(max(info["local_workspace"],info["full_workspace"]),
                                dtype=torch.uint8,device=x.device)
            ok=True
            for (v,wt),ref in zip(examples,refs):
                if plan.invoke(i,True,wt,v,full,scratch) or not torch.equal(full.view(torch.int16),ref.view(torch.int16)):
                    ok=False
                    break
                for j,weights in enumerate(wt.split(n//world)):
                    status=plan.invoke(i,False,weights,v,local,scratch)
                    if status or not torch.equal(local.view(torch.int16),ref.narrow(1,j*(n//world),n//world).view(torch.int16)):
                        ok=False
                        break
                if not ok: break
            tried.append({**info,"passed":ok})
            if ok:
                plan.index=i
                plan.k=k
                plan.workspace=torch.empty(info["local_workspace"],dtype=torch.uint8,device=x.device)
                plan.stream=torch.cuda.current_stream().cuda_stream
                return plan,{**information,"chosen":info}
        plan.close()
        return None,{**information,"reason":"no bitwise match"}
    except Exception:
        plan.close()
        raise

