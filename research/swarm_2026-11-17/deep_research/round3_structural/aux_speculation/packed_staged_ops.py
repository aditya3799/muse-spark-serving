"""Unvalidated GPU prototype: source packed NVFP4 -> quarter-valued packed BF16."""
import torch
import triton
import triton.language as tl
@triton.jit
def packed_dequant_kernel(W,S,Y,N:tl.constexpr,K:tl.constexpr,SRC_TRANSPOSE:tl.constexpr,DST_TRANSPOSE:tl.constexpr,BLOCK:tl.constexpr):
 out=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=out<N*K
 kt=out//(N*16);within=out%(N*16);nb=within//1024;rem=within%1024;lane=rem//32;inner=rem%32
 i3=inner//8;a=(inner//4)%2;b=(inner//2)%2
 if DST_TRANSPOSE:i4=b;i5=a
 else:i4=a;i5=b
 n=nb*64+i3*16+i4*8+lane//4;k=kt*16+i5*8+(lane%4)*2+inner%2
 lane_src=(n%8)*4+(k%8)//2;si3=(n%64)//16;si4=(n//8)%2;si5=(k//8)%2
 if SRC_TRANSPOSE:sa=si5;sb=si4
 else:sa=si4;sb=si5
 source_inner=((si3*2+sa)*2+sb)*2+k%2
 part=source_inner//8;sub0=source_inner%8;sub=sub0//2+4*(sub0%2)
 word=(k//16)*(N*2)+(n//64)*128+lane_src*4+part
 nibble=word*8+sub
 byte=tl.load(W+nibble//2,mask=valid,other=0).to(tl.int32);code=(byte>>((nibble%2)*4))&15;mag=code&7
 q=tl.where(mag<2,mag*.5,tl.where(mag<4,1.+(mag-2)*.5,tl.where(mag<6,mag-2.,2.*(mag-4))))
 q=(q.to(tl.int32,bitcast=True)|((code&8)<<28)).to(tl.float32,bitcast=True)
 sp=(k//16)*N+(n//64)*64+(n%8)*8+(n%64)//8
 bits8=tl.load(S+sp,mask=valid,other=0).to(tl.int32);se=(bits8>>3)&15;sm=bits8&7
 normal=((se+120)<<23)|(sm<<20);subnormal=(sm.to(tl.float32)*(1./512)).to(tl.int32,bitcast=True)
 bits=tl.where(se==0,subnormal,normal);bits=tl.where((se==15)&(sm==7),0x7fc00000,bits)|((bits8&128)<<24)
 scale=bits.to(tl.float32,bitcast=True)
 tl.store(Y+out,(q*scale*.25).to(tl.bfloat16),mask=valid)

def packed_dequant(weight,scale,output,*,n,k,source_transpose,target_transpose):
    # Exact metadata capability guards; no inference from timing buckets.
    assert weight.dtype==torch.int32 and scale.dtype==torch.float8_e4m3fn
    assert output.dtype==torch.bfloat16 and output.is_contiguous()
    assert n%64==0 and k%64==0 and weight.numel()*8==n*k and scale.numel()*16==n*k and output.numel()==n*k
    assert weight.is_contiguous() and scale.is_contiguous()
    assert weight.device==scale.device==output.device
    assert isinstance(source_transpose,bool) and isinstance(target_transpose,bool)
    # Caller owns output until all grouped/indexed GEMM consumers complete.
    packed_dequant_kernel[(triton.cdiv(n*k,4096),)](weight.view(torch.uint8),scale.view(torch.uint8),output,n,k,source_transpose,target_transpose,4096)
