"""Optional inference-only complete-work tuner. No fixed model, shapes, or algorithm IDs.

Call prepare() during warmup, after loading the replicated bias-free projection. Calibrate
with fresh sample tensors on the appropriate stream. An unsupported/cold shape uses the
original PyTorch layer. Set empirical=True only when accepting finite-test equivalence.
"""
import statistics
import time

import torch
from torch.nn import functional as F

from native_abi import load_lib, describe
from runtime_projection import native_workspace


def policy():
    mm=torch.backends.cuda.matmul
    return (str(torch.backends.cuda.preferred_blas_library()),mm.allow_bf16_reduced_precision_reduction,
            getattr(mm,"allow_bf16_reduced_precision_reduction_split_k",None),
            torch.are_deterministic_algorithms_enabled(),torch.is_autocast_enabled("cuda"))


class CompleteProjection:
    def __init__(self,layer,*,empirical=False):
        if not isinstance(layer,torch.nn.Linear) or layer.bias is not None:
            raise TypeError("expected native bias-free Linear")
        self.layer=layer
        self.empirical=empirical
        self.entries={}
        self.lib=None

    def key(self,x):
        w=self.layer.weight
        if (self.layer.training or x.ndim!=2 or w.ndim!=2 or x.dtype!=torch.bfloat16 or w.dtype!=x.dtype
            or not x.is_cuda or w.device!=x.device or x.shape[1]!=w.shape[1] or not x.is_contiguous()
            or not w.is_contiguous() or x.data_ptr()%256 or w.data_ptr()%256 or torch.is_autocast_enabled("cuda")
            or (torch.is_grad_enabled() and (x.requires_grad or w.requires_grad))):
            return None
        try:
            version=w._version
        except RuntimeError:
            # Inference tensors lack version counters. Their owner must clear
            # the tuner when loading replacement weights in place.
            version=None
        return (x.device.index,x.shape[0],w.shape[0],w.shape[1],w.data_ptr(),version,
                torch.cuda.current_stream(x.device).cuda_stream,policy())

    def invoke(self,entry,x):
        q,i,scratch=entry
        w=self.layer.weight
        out=torch.empty((x.shape[0],w.shape[0]),dtype=x.dtype,device=x.device)
        rc=self.lib.sweep_call(q,i,1,w.data_ptr(),x.data_ptr(),out.data_ptr(),
            scratch.data_ptr() if scratch.numel() else None,scratch.numel(),torch.cuda.current_stream(x.device).cuda_stream)
        return None if rc else out

    @torch.no_grad()
    def prepare(self,tune,holdout,*,max_workspace=None,max_split=None,pairs=3):
        tune,holdout=list(tune),list(holdout)
        if not tune or not holdout: return {"reason":"need distinct tuning and validation samples"}
        x=tune[0]
        key=self.key(x)
        if key is None or any(self.key(v)!=key for v in tune+holdout):
            return {"reason":"unsupported or inconsistent geometry"}
        if torch.cuda.is_current_stream_capturing() or key in self.entries:
            return {"reason":"unavailable during capture or already registered"}
        if not set(v.data_ptr() for v in tune).isdisjoint(v.data_ptr() for v in holdout):
            return {"reason":"overlapping tune and validation samples"}
        try:
            native=native_workspace()
            if native is None: return {"reason":"unknown vendor workspace"}
            if self.lib is None: self.lib=load_lib()
        except (OSError,AttributeError): return {"reason":"native probe unavailable"}
        if pairs<2 or (max_split is not None and max_split<1): raise ValueError("invalid autotuner bounds")
        current_budget=native[1]
        limit=current_budget if max_workspace is None else int(max_workspace)
        if limit<0: raise ValueError("workspace limit must be nonnegative")
        m,n,k=key[1:4]
        start=time.perf_counter()
        q=self.lib.make_sweep(m,n,n,k,n,min(current_budget,limit),limit,-1 if max_split is None else max_split)
        if not q: return {"reason":"no vendor plan"}
        scratch=torch.empty(limit,device=x.device,dtype=torch.uint8)
        def equal(a,b):
            return a is not None and torch.equal(a.view(torch.int16),b.view(torch.int16))
        def clock(call):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            y=call()
            b.record()
            b.synchronize()
            return a.elapsed_time(b)*1000
        try:
            truth=[self.layer(v) for v in tune]
            best_us=float("inf")
            best=None
            for i in range(self.lib.sweep_count(q)):
                entry=(q,i,scratch)
                if not all(equal(self.invoke(entry,v),ref) for v,ref in zip(tune,truth)):
                    continue
                sample=statistics.median(clock(lambda:self.invoke(entry,x)) for _ in range(3))
                if sample<best_us: best,best_us=i,sample
            if best is None: return {"reason":"no exact training plan"}
            required=describe(self.lib,q,best)["workspace"][1]
            live=torch.empty(required,device=x.device,dtype=torch.uint8)
            winner=(q,best,live)
            if not all(equal(self.invoke(winner,v),self.layer(v)) for v in holdout):
                return {"reason":"fresh input mismatch"}
            native_us,candidate_us,control_us=[],[],[]
            for j in range(pairs):
                v=holdout[j%len(holdout)]
                calls=[(native_us,lambda:self.layer(v)),(candidate_us,lambda:self.invoke(winner,v)),
                       (control_us,lambda:self.layer(v))]
                for order in (calls,list(reversed(calls))):
                    for dst,fn in order: dst.append(clock(fn))
            speed=statistics.median(native_us)-statistics.median(candidate_us)
            noise=abs(statistics.median(native_us)-statistics.median(control_us))
            elapsed=time.perf_counter()-start
            qualifies=speed>noise and all((native_us[j]+native_us[j+1])>(candidate_us[j]+candidate_us[j+1])
                                         for j in range(0,len(native_us),2))
            metadata={"vendor_plans":self.lib.sweep_count(q),"anchor_only":max_split is None,"admitted":qualifies,"setup_seconds":elapsed,"workspace_bytes":required,
                      "native_us":native_us,"candidate_us":candidate_us,"control_us":control_us,
                      "estimated_break_even_calls":elapsed*1e6/speed if speed>0 else None}
            if qualifies:
                self.entries[key]=(winner,native)
                q=None
            return metadata
        finally:
            if q: self.lib.drop_sweep(q)

    def __call__(self,x):
        selected=self.entries.get(self.key(x))
        if selected is None: return self.layer(x)
        entry,workspace=selected
        if native_workspace()!=workspace: return self.layer(x)
        out=self.invoke(entry,x)
        if out is None: return self.layer(x)
        if self.empirical: return out
        parent=self.layer(x)
        a,b=out.view(torch.int16),parent.view(torch.int16)
        return torch.where((a==b).all(),a,b).view(torch.bfloat16)

    def close(self):
        # Stop all graph references before destroying their underlying handles.
        for entry,_ in self.entries.values(): self.lib.drop_sweep(entry[0])
        self.entries.clear()

