# SPDX-License-Identifier: Apache-2.0
"""One bounded scratch arena, reused across layers and sequential chunks."""
import ctypes as ct
import threading
import torch
import triton
import triton.language as tl
from .muse_spark_grouped_prep import convert, gather, route, route_buffers, scatter
from .muse_spark_grouped_build import library

MAX_M = 2048
EXTRA_LIMIT = 1_380_000_000

@triton.jit
def _restore_labels(up, ids, dest, labels, W:tl.constexpr, B:tl.constexpr):
    s=tl.program_id(0)
    row=tl.load(dest+s)
    if row>=0:
        c=tl.program_id(1)*B+tl.arange(0,B)
        ix=row.to(tl.int64)*W+c
        tl.store(up+ix,tl.load(up+ix).to(tl.float32)*4.)
        if tl.program_id(1)==0:
            tl.store(labels+row,tl.load(ids+s))

def checked(rc):
    if rc:
        raise RuntimeError('grouped Hopper adapter status '+str(rc))

class Arena:
    def __init__(self, device):
        self.device=torch.device(device)
        self.stream=torch.cuda.current_stream(self.device)
        self.thread_id=threading.get_ident()
        self.forward_stream=None
        self.lib=library()
        void=ct.c_void_p
        self.lib.okc_create.argtypes=(ct.POINTER(void),ct.c_int,ct.c_int,ct.c_int,void)
        self.lib.okc_run.argtypes=(void,void,void,void,void,void,void,void)
        self.lib.okc_destroy.argtypes=(void,)
        self.lib.okc_owned_bytes.argtypes=(void,)
        self.lib.okc_owned_bytes.restype=ct.c_size_t
        for method in (self.lib.okc_create,self.lib.okc_run,self.lib.okc_destroy):
            method.restype=ct.c_int
        self.handles=[]
        try:
            for n in (8192,4096):
                handle=ct.c_void_p()
                checked(self.lib.okc_create(ct.byref(handle),32,n,4096,self.stream.cuda_stream))
                self.handles.append(handle)
            kw=dict(device=self.device)
            self.chosen=torch.arange(32,**kw,dtype=torch.int32)
            self.packed=[torch.empty(32*n*4096*9//16,**kw,dtype=torch.uint8) for n in (8192,4096)]
            rows=MAX_M*8+16*32
            self.main=torch.empty((rows,8192),**kw,dtype=torch.bfloat16)
            self.mid=torch.empty((rows,4096),**kw,dtype=torch.bfloat16)
            self.labels=torch.empty(rows,**kw,dtype=torch.int32)
            self.rbuf=route_buffers(torch.empty((MAX_M,8),**kw,dtype=torch.int32),32)
            tensors=[self.chosen,*self.packed,self.main,self.mid,self.labels,*self.rbuf]
            self.owned_bytes=sum(x.numel()*x.element_size() for x in tensors)
            self.owned_bytes+=sum(self.lib.okc_owned_bytes(h) for h in self.handles)
            if self.owned_bytes>EXTRA_LIMIT:
                raise RuntimeError('grouped scratch exceeds explicit memory budget')
            self.operands={}
        except Exception:
            self.close()
            raise

    def close(self):
        with torch.cuda.device(self.device):
            while self.handles:
                checked(self.lib.okc_destroy(self.handles.pop()))

    def prepare(self,layer):
        # All original packed coefficients/scales remain untouched and resident.
        # Both temporary stages are overwritten once per layer call, then reused by chunks.
        for j,stage in enumerate(('w13','w2')):
            meta=layer.humming_metas[stage]
            self.operands[stage]=convert(getattr(layer,meta.weight_name),
                getattr(layer,meta.weight_scale_name),self.chosen,self.packed[j],
                meta.shape_n,meta.shape_k,'wgmma' in str(meta.mma_type).lower())

    def _gemm(self,j,stage,x,y,base,padded):
        codes,words=self.operands[stage]
        checked(self.lib.okc_run(self.handles[j],
            *(t.data_ptr() for t in (codes,words,x,y,base,padded)),self.stream.cuda_stream))

    def chunk(self,layer,x,ids,defer_downscale):
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from .muse_spark_humming_scale import scale_active_rows
        m=x.size(0)
        slots=m*8
        rows=slots+16*32
        partial,prefix,dest,base,counts,padded=self.rbuf
        chunks=triton.cdiv(slots,256)
        mapping=route(ids,(partial[:chunks],prefix[:chunks],dest[:slots],base,counts,padded))
        mid=self.mid[:rows]
        main=self.main[:rows]
        labels=self.labels[:rows]
        dest,base,_,padded=mapping
        gather(x,mid,mapping)
        self._gemm(0,'w13',mid,main,base,padded)
        labels.fill_(-1)
        _restore_labels[(slots,8192//256)](main,ids,dest,labels,8192,256)
        scale_active_rows(main,labels,layer.w13_post_scale)
        silu_and_mul(main,mid,expert_ids=labels,expert_step=1)
        down=main.view(-1)[:rows*4096].view(-1,4096)
        self._gemm(1,'w2',mid,down,base,padded)
        canonical=mid[:slots]
        scatter(down,canonical,dest,restore=True)
        if not defer_downscale:
            scale_active_rows(canonical,ids,layer.w2_post_scale)
        return canonical.view(m,8,4096)

    def forward(self,layer,x,ids,defer_downscale):
        self.stream=torch.cuda.current_stream(self.device)
        self.prepare(layer)
        # The canonical output is the same contract-sized allocation as the original helper.
        out=torch.empty((x.size(0),8,4096),device=x.device,dtype=torch.bfloat16)
        for first in range(0,x.size(0),MAX_M):
            end=min(first+MAX_M,x.size(0))
            out[first:end].copy_(self.chunk(layer,x[first:end],ids[first:end],defer_downscale))
        return out,ids
