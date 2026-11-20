"""Full-K BF16 × BF16 → FP32 projection for pre-qualified prefill row shards."""
from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import os
import statistics
import sysconfig
import threading
from pathlib import Path

import torch
from sglang.srt.utils.custom_op import register_custom_op

log = logging.getLogger(__name__)
_plans = {}
_seen = set()
_lock = threading.RLock()


@functools.lru_cache(maxsize=1)
def _library():
    if os.name != "posix":
        return None
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load

        wheel = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
        roots = [wheel / "cu13", wheel / "cublas", wheel / "cuda_runtime"]
        if CUDA_HOME:
            roots.append(Path(CUDA_HOME))
        includes = [p / "include" for p in roots if (p / "include").is_dir()]
        includes.extend([p / "cccl" for p in list(includes) if (p / "cccl").is_dir()])
        if not any((p / "cublasLt.h").exists() for p in includes):
            return None
        library = next((so for p in roots for d in ("lib", "lib64") for so in
                        sorted((p / d).glob("libcublasLt.so*")) if so.is_file()), None)
        if library is None:
            return None
        src = Path(__file__).with_name("muse_target_projection.cc")
        name = "sgl_target_lt_" + hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        path = load(name, [str(src)], extra_include_paths=[str(p) for p in includes],
                    extra_cflags=["-O2"], extra_ldflags=[str(library),"-Wl,-rpath,"+str(library.parent)],
                    is_python_module=False, with_cuda=False, verbose=False)
        dll = ctypes.CDLL(path)
        ptr, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        for symbol, signature, result in (
            ("make_sweep",[integer]*5+[size,size,integer],ptr), ("sweep_count",[ptr],integer),
            ("sweep_attr",[ptr,integer,ctypes.POINTER(integer),ctypes.POINTER(size)],None),
            ("sweep_call",[ptr,integer,integer,ptr,ptr,ptr,ptr,size,ptr],integer),("drop_sweep",[ptr],None),
        ):
            fn = getattr(dll,symbol)
            fn.argtypes,fn.restype=signature,result
        return dll
    except (OSError,RuntimeError,ImportError,AttributeError) as exc:
        log.info("Target projection uses native GEMM: %s", exc)
        return None


def _native(x, weight):
    return torch.mm(x, weight.t(), out_dtype=torch.float32)


def _layout(x, weight):
    return (x.device.index, x.shape[0], weight.shape[0], weight.shape[1],
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.are_deterministic_algorithms_enabled())


def _compatible(x, weight):
    return (x.is_cuda and x.device == weight.device and x.ndim == weight.ndim == 2
            and torch.cuda.current_device() == x.device.index
            and x.dtype == weight.dtype == torch.bfloat16 and x.is_contiguous() and weight.is_contiguous()
            and x.shape[1] == weight.shape[1] and x.data_ptr() % 256 == weight.data_ptr() % 256 == 0
            and not torch.are_deterministic_algorithms_enabled())


class Plan:
    def __init__(self, dll, handle):
        self.dll, self.handle, self.index = dll, handle, None

    def run(self, index, x, weight):
        y = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=torch.float32)
        code = self.dll.sweep_call(self.handle, index, 1, weight.data_ptr(), x.data_ptr(), y.data_ptr(),
                                   0, 0, torch.cuda.current_stream(x.device).cuda_stream)
        if code:
            return None
        return y

    def close(self):
        if self.handle:
            self.dll.drop_sweep(self.handle)
            self.handle = None


def _elapsed(fn):
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    result=fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) if result is not None else float("inf")


def _calibrate(x, actual_weight):
    dll = _library()
    if dll is None:
        return None
    m, k = x.shape
    n = actual_weight.shape[0]
    # Only zero-workspace plans can execute. Discovery also queries larger
    # workspace budgets to expose valid vendor tiles before their split checks.
    cap = 32 * 1024**2
    h = dll.make_sweep(m,n,n,k,n,0,cap,4)
    if not h:
        return None
    plan=Plan(dll,h)
    try:
        candidates=[]
        for i in range(dll.sweep_count(h)):
            attr,workspace=(ctypes.c_int*10)(),(ctypes.c_uint64*3)()
            dll.sweep_attr(h,i,attr,workspace)
            if workspace[1]==0:
                candidates.append(i)
        if not candidates:
            return None
        gen=torch.Generator(device=x.device).manual_seed(919343)
        for mode in ("actual","cancellation","finite"):
            x.normal_(std=.1,generator=gen)
            if mode=="actual":
                w=actual_weight
            else:
                w=torch.empty_like(actual_weight).normal_(std=.025,generator=gen)
            if mode=="cancellation":
                # Power-of-two signed cancellation with one representable perturbation.
                xx=x.view(m,-1,2,64)
                ww=w.view(n,-1,2,64)
                xx[:,:,1].copy_(xx[:,:,0]);ww[:,:,1].copy_(ww[:,:,0]).neg_()
                ww[::3,:,1,0].mul_(1.0078125)
            if mode=="finite":
                x[:,::4]*=128
                w[:,::4]/=128
                x[::11,1::7]=-0.0
            ref=_native(x,w)
            next_candidates=[]
            for i in candidates:
                y=plan.run(i,x,w)
                if y is not None and torch.equal(y.view(torch.int32),ref.view(torch.int32)):
                    next_candidates.append(i)
            candidates=next_candidates
            if not candidates:
                return None
        # Compare the trial on the live model weight. Confirm the best choice
        # in balanced order with a duplicate native control before accepting.
        x.normal_(std=.1,generator=gen)
        ref=_native(x,actual_weight)
        passing=[]
        for i in candidates:
            y=plan.run(i,x,actual_weight)
            if y is not None and torch.equal(y.view(torch.int32),ref.view(torch.int32)):
                passing.append((statistics.median(_elapsed(lambda:plan.run(i,x,actual_weight)) for _ in range(3)),i))
        if not passing:
            return None
        best=min(passing)[1]
        native=lambda:_native(x,actual_weight)
        trial=lambda:plan.run(best,x,actual_weight)
        pairs,controls=[],[]
        for step in range(6):
            if step%2:
                c,b,a=_elapsed(native),_elapsed(trial),_elapsed(native)
            else:
                a,b,c=_elapsed(native),_elapsed(trial),_elapsed(native)
            pairs.append(a/max(b,1e-9))
            controls.append(a/max(c,1e-9))
        if min(pairs)<=1 or statistics.median(pairs)<=1.02:
            return None
        if abs(statistics.median(controls)-1)>.02:
            return None
        plan.index=best
        return plan
    finally:
        if plan.index is None:
            plan.close()


@torch.inference_mode()
def prepare(rows, weight):
    if (not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.ndim!=2 or not weight.is_contiguous()
            or rows<=0 or weight.shape[1]%128 or weight.shape[0]%128 or weight.data_ptr()%256!=0
            or torch.cuda.is_current_stream_capturing() or torch.are_deterministic_algorithms_enabled()):
        return
    with _lock, torch.cuda.device(weight.device):
        # Check the layout cache before allocating or compiling.
        key=(weight.device.index,rows,weight.shape[0],weight.shape[1],
             torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
             torch.are_deterministic_algorithms_enabled())
        if key in _seen:
            return
        _seen.add(key)
        try:
            x=torch.empty((rows,weight.shape[1]),device=weight.device,dtype=torch.bfloat16)
            if not _compatible(x,weight):
                return
            with torch.cuda.stream(torch.cuda.current_stream(weight.device)):
                plan=_calibrate(x,weight)
            if plan is not None:
                _plans[key]=plan
        except (RuntimeError,OSError,MemoryError) as exc:
            log.info("Target projection warmup uses native GEMM: %s",exc)


def _fake(x, weight):
    return x.new_empty((x.shape[0],weight.shape[0]),dtype=torch.float32)


@register_custom_op(fake_impl=_fake)
def qualified_target_row_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if _compatible(x,weight):
        plan=_plans.get(_layout(x,weight))
        if plan is not None:
            y=plan.run(plan.index,x,weight)
            if y is not None:
                return y
    return _native(x,weight)

