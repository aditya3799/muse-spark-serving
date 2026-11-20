// Research-only integer tile adapter. No floating-point arithmetic or production dispatch.
// Derived from Humming 0.1.12 process.cuh/transform.py and v139 _wgs_copy.
// The persistent source remains the scored Humming layout. This direct gather is
// an executable correctness oracle, NOT the proposed fast persistent producer.
#pragma once
#include <stdint.h>
namespace canonical_tile {
struct Address { uint32_t word, shift; };
template<bool Transpose=true>
__host__ __device__ inline Address address(uint32_t n, uint32_t k, uint32_t N) {
  uint32_t lane=(n%8)*4+(k%8)/2, i=(n%64)/8, j=(k%64)/8;
  uint32_t a=Transpose?j%2:i%2, b=Transpose?i%2:j%2;
  uint32_t tmp=((((j/2)*4+i/2)*2+a)*2+b)*2+k%2;
  uint32_t inner=tmp%8, slot=inner/2+4*(inner%2);
  return {(k/64*4+tmp/32)*(2*N)+n/64*128+lane*4+(tmp%32)/8,4*slot};
}
__host__ __device__ inline uint32_t scale_address(uint32_t n,uint32_t group,uint32_t N) {
  return group*N+n/64*64+n%8*8+n%64/8;
}
template<bool GateUp>
__host__ __device__ inline uint32_t row(uint32_t logical) {
  return GateUp ? logical/16*8+logical%8+(logical%16>=8?4096:0) : logical;
}
template<bool Mul>
__host__ __device__ inline uint32_t position(uint32_t reg,uint32_t upper,uint32_t bit) {
  uint32_t o=bit==0?6:bit==1?7:bit==2?8:15;
  if constexpr (Mul) {
    if(reg<3) return (o+16*upper+3*reg)%32;
    return (bit==0?4:bit==1?0:bit==2?1:3)+16*upper;
  } else {
    uint32_t delta=reg==0?0:reg==1?28:reg==2?22:bit==2?18:26;
    return (o+16*upper+delta)%32;
  }
}
template<bool GateUp,bool Mul,bool Transpose=true>
__device__ inline uint32_t gather_word(const uint32_t* codes,uint32_t N,uint32_t unit,uint32_t stage,uint32_t q,uint32_t word) {
  uint32_t grp=word/512, warp=word%512/128, lane=word%128/4, word_j=word%4;
  uint32_t step=stage*16+grp*4+lane%4, out=0;
  #pragma unroll
  for(uint32_t reg=0;reg<4;++reg) {
    uint32_t n=row<GateUp>(unit*256+q*64+warp*16+lane/4+(reg%2)*8);
    #pragma unroll
    for(uint32_t upper=0;upper<2;++upper) {
      uint32_t k=step*16+(reg/2)*8+word_j*2+upper;
      Address a=address<Transpose>(n,k,N);
      uint32_t code=(codes[a.word]>>a.shift)&15;
      #pragma unroll
      for(uint32_t bit=0;bit<4;++bit) out|=((code>>bit)&1)<<position<Mul>(reg,upper,bit);
    }
  }
  return out;
}
// Output is exactly 4 * (8192 code bytes + 1024 scale bytes) = 36 KiB.
// Expose only to diagnostic harness; no fallback/state ownership is changed.
template<bool GateUp,bool Mul,bool Transpose=true>
__device__ inline void diagnostic_stage_body(const uint32_t* codes,const uint8_t* scales,uint8_t* out,uint32_t N,uint32_t unit,uint32_t stage) {
  for(uint32_t iw=threadIdx.x;iw<8192;iw+=blockDim.x) {
    uint32_t q=iw/2048, word=iw%2048;
    reinterpret_cast<uint32_t*>(out+q*9216)[word]=gather_word<GateUp,Mul,Transpose>(codes,N,unit,stage,q,word);
  }
  for(uint32_t is=threadIdx.x;is<4096;is+=blockDim.x) {
    uint32_t q=is/1024, s=is%1024, h=s%2, j=s/2%4, g=s/8%8, warp=s/64%4, grp=s/256;
    uint32_t n=row<GateUp>(unit*256+q*64+warp*16+g+8*h);
    out[q*9216+8192+s]=scales[scale_address(n,stage*16+grp*4+j,N)];
  }
}
} // namespace canonical_tile
