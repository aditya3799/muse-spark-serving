// Isolated mapping/math microprobe. Not a pipelined performance kernel.
#include <cuda_bf16.h>
#include <cstdint>
constexpr int N=256, K=4096, ROWS=24, BANKS=2;
// Exact finite-scale decoder and quarter-FP4 values from independently tested native_materialize.cuh.
__device__ __forceinline__ float decode_scale(uint32_t code) {
 uint32_t exponent=(code>>3)&15u,fraction=code&7u;
 uint32_t bits=((exponent+120u)<<23)|(fraction<<20);
 if(exponent==0u)bits=__float_as_uint(__fmul_rn((float)fraction,0.001953125f));
 if(exponent==15u&&fraction==7u)bits=0x7fc00000u;
 return __uint_as_float(bits|((code&128u)<<24));
}
__device__ __forceinline__ float decode_quarter(uint32_t code) {
 uint32_t mag=code&7u;
 uint32_t bits=mag>=2u?0x3e000000u+(mag<<22):(mag==1u?0x3e000000u:0u);
 return __uint_as_float(bits|((code&8u)<<28));
}
__device__ __forceinline__ uint16_t weight_value(const uint32_t*w,const uint8_t*s,int n,int k) {
 // Literal inverse of pinned B4/A16 nonpacked-K WGMMA-transposed mini-block pack.
 uint32_t lane=(n%8)*4+(k%8)/2,i3=(n%64)/16,a=(k/8)%2,b=(n/8)%2;
 uint32_t inner=((i3*2+a)*2+b)*2+k%2;
 uint32_t sub=(inner%8)/2+4*(inner%2);
 uint32_t word=(k/16)*(N*2)+(n/64)*128+lane*4+inner/8;
 uint32_t code=(w[word]>>(4*sub))&15u;
 uint32_t sf=(k/16)*N+(n/64)*64+(n%8)*8+(n%64)/8;
 return __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(decode_quarter(code),decode_scale(s[sf]))));
}
__device__ __forceinline__ uint16_t activation(const uint16_t*x,const int32_t*slots,int compact,int k,int slots_capacity) {
 if(compact>=ROWS)return 0;
 int slot=slots[compact];
 return (slot>=0&&slot<slots_capacity)?x[(slot/8)*K+k]:0;
}
__device__ __forceinline__ uint32_t pack_bf16_pair(uint16_t a,uint16_t b){return (uint32_t)a|((uint32_t)b<<16);}
__device__ __forceinline__ void mma(uint32_t*a,uint32_t*b,float*c) {
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ void store(float v,uint16_t*out,float*raw,int slot,int n,int capacity) {
 if(slot>=0&&slot<capacity){
  raw[slot*N+n]=v;
  // Pinned NVFP4: FP32 bank reduction -> BF16 -> BF16 exponent-offset x4.
  nv_bfloat16 rounded=__float2bfloat16_rn(v);
  out[slot*N+n]=__bfloat16_as_ushort(__hmul(rounded,__float2bfloat16_rn(4.0f)));
 }
}
template<bool Swap> __device__ void run(const uint16_t*x,const uint32_t*w,const uint8_t*s,const int32_t*slots,uint16_t*out,float*raw,int capacity) {
 constexpr int TILES=Swap?ROWS/8:(ROWS+15)/16;
 float c[BANKS][TILES][4]={};
 int lane=threadIdx.x%32,nbase=blockIdx.x*(Swap?16:8);
 // Bank b visits BK*outer + 32*b + 16*iter. Keep this order before final bank0+bank1.
 for(int outer=0;outer<K;outer+=32*BANKS){
  #pragma unroll
  for(int bank=0;bank<BANKS;++bank){
   #pragma unroll
   for(int iter=0;iter<2;++iter){
    int kb=outer+32*bank+16*iter;
    uint32_t wa[4],wb[2];
    if constexpr(Swap){
     #pragma unroll
     for(int r=0;r<4;++r){int n=nbase+lane/4+(r%2)*8,k=kb+(lane%4)*2+(r/2)*8;wa[r]=pack_bf16_pair(weight_value(w,s,n,k),weight_value(w,s,n,k+1));}
    }else{
     #pragma unroll
     for(int r=0;r<2;++r){int n=nbase+lane/4,k=kb+(lane%4)*2+r*8;wb[r]=pack_bf16_pair(weight_value(w,s,n,k),weight_value(w,s,n,k+1));}
    }
    #pragma unroll
    for(int t=0;t<TILES;++t){
     // Semi-static first tile; remaining tiles skipped only under front-packed route contract.
     if(t>0&&(slots[t*(Swap?8:16)]<0||slots[t*(Swap?8:16)]>=capacity))continue;
     uint32_t aa[4],ab[2];
     if constexpr(Swap){
      #pragma unroll
      for(int r=0;r<2;++r){int row=t*8+lane/4,k=kb+(lane%4)*2+r*8;ab[r]=pack_bf16_pair(activation(x,slots,row,k,capacity),activation(x,slots,row,k+1,capacity));}
      mma(wa,ab,c[bank][t]);
     }else{
      #pragma unroll
      for(int r=0;r<4;++r){int row=t*16+lane/4+(r%2)*8,k=kb+(lane%4)*2+(r/2)*8;aa[r]=pack_bf16_pair(activation(x,slots,row,k,capacity),activation(x,slots,row,k+1,capacity));}
      mma(aa,wb,c[bank][t]);
     }
    }
   }
  }
 }
 #pragma unroll
 for(int t=0;t<TILES;++t){
  #pragma unroll
  for(int r=0;r<4;++r){
   float v=c[0][t][r];if constexpr(BANKS==2)v=__fadd_rn(v,c[1][t][r]);
   int row=Swap?t*8+(lane%4)*2+r%2:t*16+lane/4+(r/2)*8;
   int n=Swap?nbase+lane/4+(r/2)*8:nbase+(lane%4)*2+r%2;
   if(row<ROWS)store(v,out,raw,slots[row],n,capacity);
  }
 }
}
extern "C" __global__ void swap_ab(const uint16_t*x,const uint32_t*w,const uint8_t*s,const int32_t*slots,uint16_t*out,float*raw,int capacity){run<true>(x,w,s,slots,out,raw,capacity);}
extern "C" __global__ void standard_mma(const uint16_t*x,const uint32_t*w,const uint8_t*s,const int32_t*slots,uint16_t*out,float*raw,int capacity){run<false>(x,w,s,slots,out,raw,capacity);}
extern "C" __global__ void unpack_operand(const uint32_t*w,const uint8_t*s,uint16_t*out){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx<N*K)out[idx]=weight_value(w,s,idx/K,idx%K);}
