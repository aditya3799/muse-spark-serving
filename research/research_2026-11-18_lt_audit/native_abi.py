"""ctypes bridge for the CUDA-width-independent vendor-plan enumerator."""
import ctypes as ct
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


