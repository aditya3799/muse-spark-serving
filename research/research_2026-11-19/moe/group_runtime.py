"""Experimental, SM90-only frozen-parent FFN operand adapter. Qualify parity before activation."""
import ctypes as ct
from pathlib import Path
import torch
import triton
import triton.language as tl

from group_prep import convert, gather, route, route_buffers, scatter


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


def library(shared=False):
    lib=ct.CDLL(str(Path(__file__).resolve().with_name("libgroup_shared_sm90a.so" if shared else "libgroup_g16_sm90a.so")))
    void=ct.c_void_p
    lib.okc_create.argtypes=(ct.POINTER(void),ct.c_int,ct.c_int,ct.c_int,void)
    lib.okc_run.argtypes=(void,void,void,void,void,void,void,void)
    lib.okc_destroy.argtypes=(void,)
    for method in (lib.okc_create,lib.okc_run,lib.okc_destroy):
        method.restype=ct.c_int
    return lib


def checked(rc):
    if rc:
        raise RuntimeError(f"grouped Hopper adapter status {rc} (>=1000 denotes CUTLASS)")


class GroupedFFN:
    """One layer/execution stream at a time. W13 and W2 share their transient backing stores."""
    def __init__(self,m,h,i,e,device,*,exclusive=False,shared_weights=False):
        self.device=torch.device(device)
        if self.device.type=="cuda" and self.device.index is None:
            self.device=torch.device("cuda",torch.cuda.current_device())
        if self.device.type!="cuda" or torch.cuda.get_device_capability(self.device)!=(9,0):
            raise RuntimeError("grouped native adapter requires Hopper sm90a")
        if h!=i or h%256 or not 1<=e<=32 or not m or m*8+16*e>=2**31:
            raise ValueError("requires equal H/I, aligned features, 1-32 local experts, and int32-addressable rows")
        self.shape=(m,h,i,e)
        self.stream=torch.cuda.current_stream(self.device)
        self.shared_weights=shared_weights
        self.lib=library(shared_weights)
        self.handles=[]
        with torch.cuda.device(self.device):
            try:
                for n,k in ((2*i,h),(h,i)):
                    handle=ct.c_void_p()
                    checked(self.lib.okc_create(ct.byref(handle),e,n,k,self.stream.cuda_stream))
                    self.handles.append(handle)
                kw=dict(device=self.device)
                self.selected=torch.arange(e,**kw,dtype=torch.int32)
                layout=torch.empty((m,8),**kw,dtype=torch.int32)
                self.rbuf=route_buffers(layout,e)
                rows=m*8+16*e
                if exclusive:
                    self.scratch=None
                elif shared_weights:
                    self.scratch=torch.empty(e*max(2*i*h,h*i)//64,**kw,dtype=torch.int32)
                else:
                    self.scratch=torch.empty(e*max(2*i*h,h*i)*9//16,**kw,dtype=torch.uint8)
                self.main=torch.empty((rows,2*i),**kw,dtype=torch.bfloat16)
                self.mid=torch.empty((rows,h),**kw,dtype=torch.bfloat16)
                self.labels=torch.empty(rows,**kw,dtype=torch.int32)
            except Exception:
                self.close()
                raise

    def close(self):
        with torch.cuda.device(self.device):
            while self.handles:
                checked(self.lib.okc_destroy(self.handles.pop()))

    def _gemm(self,which,stage,layer,x,y,base,padded):
        from humming import dtypes
        meta=layer.humming_metas[stage]
        if (meta.b_dtype!=dtypes.float4e2m1 or meta.bs_dtype!=dtypes.float8e4m3 or
            meta.should_apply_bs_on_c or meta.use_fused_e8m0_scale or meta.weight_scale_group_size!=16
            or meta.shape_k!=x.size(1) or meta.shape_n!=y.size(1)):
            raise ValueError("incompatible frozen Humming layer format")
        w=getattr(layer,meta.weight_name)
        sf=getattr(layer,meta.weight_scale_name)
        e=self.shape[3]
        n,k=y.size(1),x.size(1)
        if w.numel()*w.element_size()!=e*n*k//2 or sf.numel()*sf.element_size()!=e*n*k//16:
            raise ValueError("invalid frozen weight/scale byte count")
        if self.shared_weights:
            if "wgmma" not in str(meta.mma_type).lower() or hasattr(layer,"_exclusive_cutlass"):
                raise ValueError("shared Hopper coefficients require the original WGMMA mini-tile")
            codes=w
            if hasattr(layer,"_shared_scale_words"):
                if sf.dtype!=torch.int32 or sf.shape!=(e,k//64,n):
                    raise ValueError("unexpected shared group scale word layout")
                words=sf
            else:
                from shared_scales import words_from_native
                if self.scratch is None:
                    raise ValueError("shared scales must be installed at setup")
                words=words_from_native(sf,self.selected,self.scratch,n,k)
        elif hasattr(layer,"_exclusive_cutlass"):
            if w.dtype!=torch.uint8 or sf.dtype!=torch.int32 or w.shape!=(e,n,k//2) or sf.shape!=(e,k//64,n):
                raise ValueError("unexpected exclusive packed layout")
            codes,words=w,sf
        else:
            if self.scratch is None:
                raise ValueError("configured for exclusive weights but layer was not installed")
            codes,words=convert(w,sf,self.selected,self.scratch,n,k,"wgmma" in str(meta.mma_type).lower())
        checked(self.lib.okc_run(self.handles[which],*(t.data_ptr() for t in (codes,words,x,y,base,padded)),
                                 self.stream.cuda_stream))

    def forward(self,layer,x,local_ids,*,defer_downscale=False):
        """Return the incumbent (M,top8,H) raw expert outputs and its local IDs, aliasing workspace."""
        from sglang.kernels.ops.activation.activation import silu_and_mul
        from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
        max_m,h,i,e=self.shape
        m=x.size(0)
        if m<1 or m>max_m:
            raise ValueError("requested rows exceed the shared stream arena")
        if (not self.handles or torch.cuda.current_device()!=self.device.index or
            torch.cuda.current_stream(self.device)!=self.stream or
            x.device!=self.device or local_ids.device!=self.device or x.shape!=(m,h) or
            x.dtype!=torch.bfloat16 or not x.is_contiguous() or local_ids.shape!=(m,8)):
            raise ValueError("invalid stream, BF16 rank input, or local route shape")
        slots=m*8
        rows=slots+16*e
        chunks=triton.cdiv(slots,256)
        partial,prefix,dest,base,counts,padded=self.rbuf
        mp=route(local_ids,(partial[:chunks],prefix[:chunks],dest[:slots],base,counts,padded))
        mid=self.mid[:rows]
        main=self.main[:rows]
        labels=self.labels[:rows]
        dest,base,_,padded=mp
        gather(x,mid,mp)
        self._gemm(0,"w13",layer,mid,main,base,padded)
        labels.fill_(-1)
        _restore_labels[(dest.numel(),2*i//256)](main,local_ids,dest,labels,2*i,256)
        scale_active_rows(main,labels,layer.w13_post_scale)
        silu_and_mul(main,mid,expert_ids=labels,expert_step=1)
        down=main.view(-1)[:rows*h].view(-1,h)
        self._gemm(1,"w2",layer,mid,down,base,padded)
        canon=mid[:m*8]
        scatter(down,canon,dest,restore=True)
        if not defer_downscale:
            scale_active_rows(canon,local_ids,layer.w2_post_scale)
        return canon.view(m,8,h),local_ids
