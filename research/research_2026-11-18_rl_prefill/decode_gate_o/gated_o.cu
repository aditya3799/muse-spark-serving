#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace {
namespace wm = nvcuda::wmma;
using bf = __nv_bfloat16;
constexpr int kMaxWidth = 1024;
constexpr int kWarps = 4;
constexpr int kWmma = 16;

template<bool kInspect>
__global__ void gated_projection(const bf* __restrict__ x, const bf* __restrict__ gate,
                                  const bf* __restrict__ w, bf* __restrict__ out,
                                  bf* __restrict__ gated, int rows, int n, int k,
                                  int gate_pitch, float eps) {
  // A complete normalized input tile is reused by all output-column warps.
  __shared__ __align__(32) bf a[kWmma][kMaxWidth];
  __shared__ __align__(32) float results[kWarps][kWmma][kWmma];
  const int tid=threadIdx.x;
  const int sublane=tid%16;
  const int heads=k/64;
  for (int row_head=tid/16; row_head<kWmma*heads; row_head+=blockDim.x/16) {
    const int row=row_head/heads;
    const int col=(row_head%heads)*64+sublane*4;
    float xv[4], gv[4], ss=0.f;
    #pragma unroll
    for (int i=0;i<4;++i) {
      xv[i]=row<rows ? __bfloat162float(x[row*k+col+i]) : 0.f;
      gv[i]=row<rows ? __bfloat162float(gate[row*gate_pitch+col+i]) : 0.f;
      ss=fmaf(xv[i],xv[i],ss);
    }
    #pragma unroll
    for (int mask=8;mask>=1;mask>>=1) ss+=__shfl_xor_sync(0xffffffff,ss,mask,16);
    const float inv=rsqrtf(ss/64.f+eps);
    #pragma unroll
    for (int i=0;i<4;++i) {
      const float sigmoid=__fdividef(1.f,1.f+__expf(-gv[i]));
      const bf val=__float2bfloat16_rn((xv[i]*inv)*sigmoid);
      a[row][col+i]=val;
      if (kInspect && blockIdx.x==0 && row<rows) gated[row*k+col+i]=val;
    }
  }
  __syncthreads();
  const int warp=tid/32, lane=tid%32;
  const int out_col=blockIdx.x*(kWarps*kWmma)+warp*kWmma;
  wm::fragment<wm::matrix_a,16,16,16,bf,wm::row_major> lhs;
  wm::fragment<wm::matrix_b,16,16,16,bf,wm::col_major> rhs;
  wm::fragment<wm::accumulator,16,16,16,float> acc;
  wm::fill_fragment(acc,0.f);
  for (int ki=0;ki<k;ki+=16) {
    wm::load_matrix_sync(lhs,&a[0][ki],kMaxWidth);
    wm::load_matrix_sync(rhs,w+size_t(out_col)*k+ki,k);
    wm::mma_sync(acc,lhs,rhs,acc);
  }
  wm::store_matrix_sync(&results[warp][0][0],acc,kWmma,wm::mem_row_major);
  __syncwarp();
  for(int i=lane;i<rows*kWmma;i+=32)
    out[(i/kWmma)*n+out_col+i%kWmma]=__float2bfloat16_rn(results[warp][i/kWmma][i%kWmma]);
}
}
extern "C" int launch_gated_o(const void* x,const void* gate,const void* w,void* out,
                             void* inspect,int rows,int n,int k,int gate_pitch,float eps,void* stream) {
  if (rows<1 || rows>16 || k<=0 || k>kMaxWidth || k%64 || n<=0 || n%64 || gate_pitch<k) return cudaErrorInvalidValue;
  const auto grid=dim3(n/(kWarps*kWmma));
  auto st=static_cast<cudaStream_t>(stream);
  if (inspect) gated_projection<true><<<grid,kWarps*32,0,st>>>((const bf*)x,(const bf*)gate,(const bf*)w,(bf*)out,(bf*)inspect,rows,n,k,gate_pitch,eps);
  else gated_projection<false><<<grid,kWarps*32,0,st>>>((const bf*)x,(const bf*)gate,(const bf*)w,(bf*)out,nullptr,rows,n,k,gate_pitch,eps);
  return cudaGetLastError();
}

