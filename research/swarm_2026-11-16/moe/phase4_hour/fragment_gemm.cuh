// Controlled MMA integration only. Same FP32 reduction loop in both variants.
#pragma once
#include "canonical_fragment.cuh"
#include "../phase3/canonical_tile_adapter.cuh"
namespace fragment_gemm {
template<bool Direct>
__device__ inline void run(const uint32_t* codes,const uint8_t* scales,const nv_bfloat16* x,float* accum,nv_bfloat16* raw,nv_bfloat16* final,uint32_t M,uint32_t N,uint32_t K,nv_bfloat16 global) {
 uint32_t lane=threadIdx.x,nb=blockIdx.x*16,mb=blockIdx.y*8;
 uint32_t n=nb+lane/4,t=mb+lane/4;float c[4]={0,0,0,0};
 for(uint32_t kk=0;kk<K;kk+=16){
  auto addr=canonical_tile::address<true>(n,kk+(lane%4)*2,N);
  uint32_t q=codes[addr.word],s0=scales[canonical_tile::scale_address(n,kk/16,N)],s1=scales[canonical_tile::scale_address(n+8,kk/16,N)],a[4];
  if constexpr(Direct) fragment::decode<0>(q,0,s0,s1,a);else fragment::pinned(q,s0,s1,a);
  uint32_t b0=0,b1=0;
  if(t<M){b0=*reinterpret_cast<const uint32_t*>(x+t*K+kk+(lane%4)*2);b1=*reinterpret_cast<const uint32_t*>(x+t*K+kk+8+(lane%4)*2);}
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
 }
 #pragma unroll
 for(int r=0;r<4;++r){uint32_t row=nb+lane/4+(r/2)*8,col=mb+(lane%4)*2+r%2;if(col<M){
  uint32_t ix=col*N+row;float value=c[r]*4.0f;accum[ix]=value;auto rounded=__float2bfloat16_rn(value);raw[ix]=rounded;final[ix]=__hmul(rounded,global);
 }}
}
}
