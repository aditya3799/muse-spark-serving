"""Experimental prologue. None means complete native fallback before cache writes.
Eager qualification writes only scratch buffers. Importing activates nothing.
"""
from __future__ import annotations
import logging,math,time
import torch
import triton
from .kernel import fusion18_kernel,conversion_probe
logger=logging.getLogger(__name__)

class Fa3Prologue:
    def __init__(self, *, calibration_budget_bytes=16*1024**2, prefer_native=True):
        self.calibration_budget_bytes=int(calibration_budget_bytes)
        self.prefer_native=bool(prefer_native)
        self.codecs={};self.shapes={};self.receipts=[];self.last_reason='not_called'

    def _reject(self,reason):
        self.last_reason=reason
        return None

    def guard(self,qkv,positions,rope,k,v,locations,*,head_dim,num_q_heads,num_kv_heads,eps,
              k_scale=None,v_scale=None,reference_is_triton=False,compatible_pool=False):
        if not reference_is_triton:return 'native_backend'
        if not compatible_pool:return 'pool_ownership'
        if k_scale is not None or v_scale is not None:return 'cache_scales'
        if (head_dim,num_q_heads,num_kv_heads)!=(64,16,2):return 'head_geometry'
        if not math.isfinite(float(eps)) or eps<0:return 'epsilon'
        if qkv.device.type!='cuda' or qkv.dtype!=torch.bfloat16:return 'input_dtype_device'
        if qkv.ndim!=2 or qkv.shape[1]!=2304 or qkv.shape[0]<=0:return 'input_shape'
        if qkv.stride(1)!=1 or qkv.stride(0)<2304:return 'input_stride'
        device=qkv.device;m=qkv.shape[0]
        if device.index!=torch.cuda.current_device():return 'current_device_mismatch'
        if any(t.device!=device for t in (k,v,locations)):return 'device_mismatch'
        if k.dtype!=torch.float8_e4m3fn or v.dtype!=k.dtype:return 'cache_dtype'
        if k.ndim!=2 or v.shape!=k.shape or k.shape[1]!=128 or k.shape[0]<=0:return 'cache_shape'
        if not k.is_contiguous() or not v.is_contiguous():return 'cache_stride'
        if locations.ndim!=1 or locations.numel()!=m or locations.dtype not in (torch.int32,torch.int64):return 'locations_shape_dtype'
        if not locations.is_contiguous():return 'locations_stride'
        if (positions is None)!=(rope is None):return 'rope_pair'
        if positions is not None:
            if positions.device!=device or rope.device!=device:return 'rope_device'
            if positions.shape!=(m,) or positions.dtype!=torch.int64 or not positions.is_contiguous():return 'positions'
            if rope.ndim!=2 or rope.shape[1]!=64 or rope.dtype!=torch.float32 or not rope.is_contiguous():return 'rope_cache'
        return None

    @staticmethod
    def _block(m):
        rows=m*18
        return 4 if rows<1024 else 8 if rows<8192 else 16 if rows<65536 else 32

    def _launch(self,qkv,positions,rope,k,v,locations,eps,codec):
        m=qkv.shape[0];block=self._block(m)
        out=torch.empty((m,1024),device=qkv.device,dtype=torch.uint8)
        fusion18_kernel[(triton.cdiv(m*18,block),)](
            qkv,out,qkv,positions if positions is not None else qkv,rope if rope is not None else qkv,
            k.view(torch.uint8),v.view(torch.uint8),locations,float(eps),qkv.stride(0),m*18,k.shape[0],
            16,2,64,64,block,positions is not None,True,True,False,NATIVE=codec=='native',num_warps=4,debug=True)
        return out.view(torch.float8_e4m3fn)

    def prepare_codec(self,device):
        index=device.index if device.index is not None else torch.cuda.current_device()
        if index in self.codecs:return self.codecs[index]
        if torch.cuda.is_current_stream_capturing():return None
        started=time.perf_counter()
        bits=torch.arange(65536,device=device,dtype=torch.int32).short().view(torch.bfloat16)
        expected=bits.to(torch.float8_e4m3fn).view(torch.uint8)
        result=torch.empty_like(expected);attempts=[];selected=None
        candidates=['native','software'] if self.prefer_native and torch.cuda.get_device_capability(device)>=(9,0) else ['software']
        for codec in candidates:
            conversion_probe[(512,)](bits,result,65536,NATIVE=codec=='native')
            exact=torch.equal(expected,result)
            attempts.append(dict(codec=codec,exact=exact,wrong=int((expected!=result).sum())))
            if exact:selected=codec;break
        self.codecs[index]=selected
        receipt=dict(kind='codec',device=index,torch=torch.__version__,attempts=attempts,selected=selected,warmup_ms=(time.perf_counter()-started)*1000)
        self.receipts.append(receipt);logger.info('MUSE_FA3_PROLOGUE %s',receipt)
        return selected

    def _qualify(self,key,qkv,positions,rope,eps,codec,reference):
        m=qkv.shape[0];estimate=m*8192+262144
        if estimate>self.calibration_budget_bytes:return False
        started=time.perf_counter()
        scratch=[torch.zeros((m+1,128),device=qkv.device,dtype=torch.float8_e4m3fn) for _ in range(2)]
        loc=torch.arange(1,m+1,device=qkv.device,dtype=torch.int64)
        q,k=reference(qkv,positions,rope)
        expected=[q.to(torch.float8_e4m3fn),k.to(torch.float8_e4m3fn),qkv[:,1152:1280].to(torch.float8_e4m3fn)]
        actual=self._launch(qkv,positions,rope,*scratch,loc,eps,codec)
        checks=[torch.equal(actual.view(torch.uint8),expected[0].view(torch.uint8)),
                torch.equal(scratch[0][1:].view(torch.uint8),expected[1].view(torch.uint8)),
                torch.equal(scratch[1][1:].view(torch.uint8),expected[2].view(torch.uint8)),
                all(bool((x[0].view(torch.uint8)==0).all()) for x in scratch)]
        accepted=all(checks)
        receipt=dict(kind='shape',key=list(key),exact=accepted,checks=checks,estimated_scratch_bytes=estimate,warmup_ms=(time.perf_counter()-started)*1000)
        self.receipts.append(receipt);logger.info('MUSE_FA3_PROLOGUE %s',receipt)
        return accepted

    def try_run(self,qkv,positions,rope,k,v,locations,*,reference,head_dim=64,num_q_heads=16,num_kv_heads=2,
                eps=1e-5,k_scale=None,v_scale=None,reference_is_triton=False,compatible_pool=False):
        reason=self.guard(qkv,positions,rope,k,v,locations,head_dim=head_dim,num_q_heads=num_q_heads,num_kv_heads=num_kv_heads,
                          eps=eps,k_scale=k_scale,v_scale=v_scale,reference_is_triton=reference_is_triton,compatible_pool=compatible_pool)
        if reason:return self._reject(reason)
        codec=self.prepare_codec(qkv.device)
        if codec is None:return self._reject('codec_unqualified')
        key=(qkv.device.index,qkv.shape[0],qkv.stride(0),positions is not None,float(eps),codec)
        if key not in self.shapes:
            if torch.cuda.is_current_stream_capturing():return self._reject('shape_unqualified_in_capture')
            self.shapes[key]=self._qualify(key,qkv,positions,rope,eps,codec,reference)
        if not self.shapes[key]:return self._reject('shape_rejected')
        self.last_reason='enabled'
        return self._launch(qkv,positions,rope,k,v,locations,eps,codec)
