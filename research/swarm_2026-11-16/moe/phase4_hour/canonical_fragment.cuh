// Isolated canonical-fragment integration. Pinned Humming arithmetic remains reference.
#pragma once
#include <humming/arith/mainloop_arith.cuh>
namespace fragment {
struct Shape {static constexpr uint32_t M=16,N=32,K=64;};
struct Op {using ValTypeC=float;using MmaShape=Shape;static constexpr uint32_t kCTypeBits=32;static constexpr bool kNativeMixed=false;};
struct Ctx {
 using MmaOpClass=Op;using BlockShape=Shape;using WarpShape=Shape;
 using ElementA=BFloat16;using ElementB=Float4E2M1;using ElementBS=Float8E4M3;using ElementC=BFloat16;
 static constexpr bool kUseWgmma=true,kUseMxmma=false,kHasZeroPoint=false,kIsFpZeroPoint=false,kUseIntWeightScale=false,kUseFusedE8m0Scale=false,kIsGroupWeightScale=true,kIsBlockWeightScale=false,kIsChannelWeightScale=false,kUsePackedKLayout=false;
 static constexpr uint32_t kInputScaleGroupSize=0,kWeightScaleGroupSize=16,kPackedKFactor=1,kWarpIters=4;
};
constexpr uint2 offset=get_mainloop_exp_offset<BFloat16,Float4E2M1,Float8E4M3,false,false,false,true,false>();
static_assert(offset.x==124 && offset.y==120);
static_assert(get_epilogue_exp_offset<BFloat16,Float4E2M1,BFloat16,Float8E4M3,false,false,false,true,false>().x==2);
__device__ inline uint32_t mul(uint32_t a,uint32_t b) {auto v=__hmul2(*reinterpret_cast<nv_bfloat162*>(&a),*reinterpret_cast<nv_bfloat162*>(&b));return *reinterpret_cast<uint32_t*>(&v);}
__device__ inline uint32_t scale_pair(uint32_t s) {
 uint32_t raw=fp_to_fp<Float8E4M3,BFloat16>((s*0x00010001u)<<8);
 auto f=prepare_exp_scale_factor<nv_bfloat162,offset.y>();
 return mul(raw,*reinterpret_cast<uint32_t*>(&f));
}
__device__ inline uint32_t direct_pair(uint32_t q,uint32_t shift,uint32_t scale) {
 uint32_t raw=fp_to_fp<Float4E2M1,BFloat16>(q<<(12-shift));
 auto f=prepare_exp_scale_factor<nv_bfloat162,offset.x>();
 return mul(mul(raw,*reinterpret_cast<uint32_t*>(&f)),scale);
}
__device__ inline void pinned(uint32_t q,uint32_t s0,uint32_t s1,uint32_t (&out)[4]) {
 MainloopArithmetic<Ctx> ar;
 uint32_t packed=s0|(s1<<8)|(s0<<16)|(s1<<24);
 ar.bs[0][0]=packed;ar.bs[0][1]=packed;
 uint32_t qb[4]={q,0,0,0},zp[4]={0,0,0,0};
 dequant<Float4E2M1,BFloat16,false,false>(qb,out,0,zp);
 ar.may_apply_bs_and_zp_on_b(out,0,0);
}
template<int Mode>
__device__ inline void decode(uint32_t q0,uint32_t q1,uint32_t s0,uint32_t s1,uint32_t (&out)[4]) {
 uint32_t a=scale_pair(s0),b=scale_pair(s1);
 #pragma unroll
 for(int r=0;r<4;++r) {
  uint32_t q=Mode==0?q0:(r%2?q1:q0);
  uint32_t shift=Mode==0?r*4:((Mode-1)*4+(r/2)*8);
  out[r]=direct_pair(q,shift,r%2?b:a);
 }
}
template<int Mode>
__device__ inline void reference(uint32_t q0,uint32_t q1,uint32_t s0,uint32_t s1,uint32_t (&out)[4]) {
 uint32_t a[4],b[4];
 if constexpr(Mode==0) {pinned(q0,s0,s1,out);}
 else {pinned(q0,s0,s0,a);pinned(q1,s1,s1,b);
  #pragma unroll
  for(int r=0;r<4;++r) out[r]=r%2?b[(Mode-1)+(r/2)*2]:a[(Mode-1)+(r/2)*2];
 }
}
template<int Mode>
__device__ inline void run(const uint32_t* inputs,uint32_t* actual,uint32_t* expected,uint32_t count) {
 uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 uint32_t q0=inputs[4*i],q1=inputs[4*i+1],s0=inputs[4*i+2],s1=inputs[4*i+3],a[4],b[4];
 decode<Mode>(q0,q1,s0,s1,a);reference<Mode>(q0,q1,s0,s1,b);
 #pragma unroll
 for(int r=0;r<4;++r){actual[4*i+r]=a[r];expected[4*i+r]=b[r];}
}
}
