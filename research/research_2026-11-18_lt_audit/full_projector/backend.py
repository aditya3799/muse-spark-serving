"""ctypes bridge for the CUDA-width-independent vendor-plan enumerator."""
import ctypes as ct
from functools import lru_cache
import torch
from pathlib import Path

HERE=Path(__file__).resolve().parent
NAMES="id tile split reduction swizzle custom stages inner cluster source".split()

def load_lib():
    lib = ct.CDLL(str(HERE / "sweep.so"))
    ptr, num, size = ct.c_void_p, ct.c_int, ct.c_uint64
    for name, sig, ret in [
        ("make_sweep", [num]*5+[size,size,num], ptr), ("sweep_count", [ptr], num),
        ("sweep_attr", [ptr,num,ct.POINTER(num),ct.POINTER(size)], None),
        ("sweep_call", [ptr,num,num,ptr,ptr,ptr,ptr,size,ptr], num),
        ("drop_sweep", [ptr], None),
    ]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = sig, ret
    return lib


def describe(lib, plan, i):
    attr, work = (ct.c_int*10)(), (ct.c_uint64*3)()
    lib.sweep_attr(plan, i, attr, work)
    return dict(**dict(zip(NAMES, attr)), workspace=list(work))



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


