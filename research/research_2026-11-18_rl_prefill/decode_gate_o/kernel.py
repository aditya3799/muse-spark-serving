"""Research binding for the fused gate + full rank-local O projection."""
import ctypes as C
from functools import lru_cache
from pathlib import Path
import torch

P=C.c_void_p
I=C.c_int

@lru_cache
def library():
    lib=C.CDLL(str(Path(__file__).with_name("gated_o.so")))
    op=lib.launch_gated_o
    op.argtypes=[P,P,P,P,P,I,I,I,I,C.c_float,P]
    op.restype=I
    return op

def eligible(x,gate,weight,head_dim,sm_count):
    if gate is None or x.ndim!=2 or weight.ndim!=2 or gate.ndim!=2:
        return False
    m,k=x.shape
    return (head_dim==64 and 0<m<=16 and 0<k<=1024 and k%64==0
            and k//64>=8 and (k//64)%8==0 and weight.shape[1]==k and weight.shape[0]>0 and weight.shape[0]%64==0
            and gate.shape==x.shape and gate.stride(1)==1 and gate.stride(0)>=k
            and x.dtype==gate.dtype==weight.dtype==torch.bfloat16 and x.device==gate.device==weight.device
            and x.is_contiguous() and weight.is_contiguous()
            and m*k*2//8<=sm_count*256)

def project(x,gate,weight,eps,inspect=False):
    assert x.is_cuda and eligible(x,gate,weight,64,torch.cuda.get_device_properties(x.device).multi_processor_count)
    out=torch.empty((x.shape[0],weight.shape[0]),device=x.device,dtype=x.dtype)
    debug=torch.empty_like(x) if inspect else None
    state=library()(x.data_ptr(),gate.data_ptr(),weight.data_ptr(),out.data_ptr(),debug.data_ptr() if inspect else None,
                    x.shape[0],weight.shape[0],x.shape[1],gate.stride(0),eps,torch.cuda.current_stream(x.device).cuda_stream)
    if state:raise RuntimeError("CUDA gated projection launch failed with status "+str(state))
    return (out,debug) if inspect else out


