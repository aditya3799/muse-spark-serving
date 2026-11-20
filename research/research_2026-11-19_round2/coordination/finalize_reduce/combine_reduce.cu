#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
using BF=__nv_bfloat16;
struct Args {
 const BF *x,*weight,*scale,*peer,*local;
 const float *scores,*inv;
 const int32_t *ids;
 BF *out,*debug_local;
 int rows,hidden,rank;
};
__device__ __forceinline__ uint32_t encode_pair(uint32_t u) {
 // Match stock LamportTrait<bf16_t,8,4>: a zero PAIR becomes 0x00008000.
 return u==0 ? 0x00008000u : u;
}
__device__ __forceinline__ uint32_t pair_bits(BF a,BF b) {
 return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b))<<16);
}
__device__ __forceinline__ float unpack(uint32_t p,int lane) {
 return __bfloat162float(__ushort_as_bfloat16(uint16_t(p>>(lane*16))));
}
template<bool Fused,bool HasWeight,bool WeightFirst,bool HasScale>
__global__ void kernel(Args a) {
 const int token=blockIdx.x, col=blockIdx.y*256+threadIdx.x*8;
 if(col>=a.hidden)return;
 BF local[8];
 if constexpr(Fused) {
  float acc[8]={0,0,0,0,0,0,0,0};
#pragma unroll
  for(int k=0;k<8;k++) {
   const int slot=token*8+k,id=a.ids[slot];
   if(id>=0) {
    const float inv=a.inv[slot],score=a.scores[slot];
#pragma unroll
    for(int j=0;j<8;j++) {
     float x=__bfloat162float(a.x[int64_t(slot)*a.hidden+col+j]);
     if constexpr(HasScale)x=__bfloat162float(__float2bfloat16_rn(x*__bfloat162float(a.scale[id])));
     float y=x;
     if constexpr(HasWeight && WeightFirst)y=(x*__bfloat162float(a.weight[col+j]))*inv;
     else if constexpr(HasWeight)y=(x*inv)*__bfloat162float(a.weight[col+j]);
     else y=x*inv;
     acc[j]=fmaf(y,score,acc[j]);
    }
   }
  }
#pragma unroll
  for(int j=0;j<8;j++) {
   local[j]=__float2bfloat16_rn(acc[j]);
   if(a.debug_local)a.debug_local[int64_t(token)*a.hidden+col+j]=local[j];
  }
 } else {
#pragma unroll
  for(int j=0;j<8;j++)local[j]=a.local[int64_t(token)*a.hidden+col+j];
 }
#pragma unroll
 for(int pair=0;pair<4;pair++) {
  float low=0,high=0;
#pragma unroll
  for(int rank=0;rank<8;rank++) {
   uint32_t bits;
   if(rank==a.rank)bits=pair_bits(local[2*pair],local[2*pair+1]);
   else bits=*reinterpret_cast<const uint32_t*>(a.peer+(int64_t(rank)*a.rows+token)*a.hidden+col+2*pair);
   bits=encode_pair(bits);
   if(rank==0){low=unpack(bits,0);high=unpack(bits,1);}
   else {low=__fadd_rn(low,unpack(bits,0));high=__fadd_rn(high,unpack(bits,1));}
  }
  a.out[int64_t(token)*a.hidden+col+pair*2]=__float2bfloat16_rn(low);
  a.out[int64_t(token)*a.hidden+col+pair*2+1]=__float2bfloat16_rn(high);
 }
}
extern "C" size_t args_size(){return sizeof(Args);}
extern "C" int run(int fused,int weight,int first,int scale,Args *a,cudaStream_t stream){
 if(a->rows<1 || a->hidden%256 || a->rank<0 || a->rank>=8)return -1;
 dim3 grid(a->rows,a->hidden/256);dim3 block(32);
 if(!fused)kernel<false,false,false,false><<<grid,block,0,stream>>>(*a);
 else {
#define CASE(W,F,S) if(weight==W && first==F && scale==S)kernel<true,W,F,S><<<grid,block,0,stream>>>(*a);
 CASE(0,0,0) CASE(0,0,1) CASE(0,1,0) CASE(0,1,1)
 CASE(1,0,0) CASE(1,0,1) CASE(1,1,0) CASE(1,1,1)
#undef CASE
 }
 return int(cudaGetLastError());
}
