// Research-only exact arithmetic + native one-shot push protocol.
// No serving hook or distributed hardware claim.
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/distributed/communicator.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
namespace sglang {
using BF=__nv_bfloat16;
struct Args {
 const BF *x,*weight,*scale,*peer,*local;
 const float *scores,*inv;
 const int32_t *ids;
 BF *out,*debug_local;
 int rows,hidden,rank;
};
struct FinalizeParams {
 Args a;
 void* output;
 uint32_t num_vecs,rank;
 device::distributed::PushWorkSpace<8> ws;
};
struct ProtoArgs {
 Args ranks[8];
 uint8_t* workspaces[8];
 device::distributed::Counter* counters[8];
 uint32_t slot_bytes,controllers;
};
template<bool Fused,bool HasScale,class V>
__device__ __forceinline__ void fill_vec(V& vec,const Args& a,uint32_t vid) {
 if constexpr(!Fused) { vec.load(a.local,vid); }
 else {
  const uint32_t token=(vid*8)/a.hidden,col=(vid*8)%a.hidden;
  BF local[8];float acc[8]={0,0,0,0,0,0,0,0};
#pragma unroll
  for(int k=0;k<8;k++) {
   const int slot=token*8+k,id=a.ids[slot];
   if(id>=0) {
    const float inv=a.inv[slot],score=a.scores[slot];
#pragma unroll
    for(int j=0;j<8;j++) {
     float x=__bfloat162float(a.x[int64_t(slot)*a.hidden+col+j]);
     if constexpr(HasScale)x=__bfloat162float(__float2bfloat16_rn(x*__bfloat162float(a.scale[id])));
     const float y=(x*__bfloat162float(a.weight[col+j]))*inv;
     acc[j]=fmaf(y,score,acc[j]);
    }
   }
  }
#pragma unroll
  for(int j=0;j<8;j++) {
   local[j]=__float2bfloat16_rn(acc[j]);
   if(a.debug_local)a.debug_local[int64_t(token)*a.hidden+col+j]=local[j];
  }
  static_assert(sizeof(vec)==sizeof(local));
  memcpy(vec.data(),local,sizeof(local));
 }
}
template<bool Fused,bool HasScale>
__device__ __forceinline__ void transport_body(const FinalizeParams& params) {
 using T=bf16_t;
 constexpr uint32_t kWorldSize=8;
 constexpr bool kUsePDL=false;

  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  using Lamport = distributed::LamportTrait<T, kVecSize * 2, /*kAtom=*/4>;
  const auto r = params.rank;
  const auto num_vecs = params.num_vecs;
  const auto num_threads = blockDim.x * gridDim.x;
  // Round-robin warps to blocks rather than giving each block a contiguous run.
  // The grid is pinned to the counter array, so once `num_vecs` stops filling
  // `gridDim * blockDim` a block-major index leaves the tail CTAs with nothing
  // to do; that happens over a whole 2x band of sizes, between the point where
  // `choose_block_size` gives up on 512 and the point where 1024 threads fill
  // the grid again.
  const auto warp_in_block = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x + gridDim.x * warp_in_block;
  const auto global_tid = global_warp_id * kWarpThreads + lane_id;
  PDLWaitPrimary<kUsePDL>();
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};

  // push to peer
  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }

  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    fill_vec<Fused, HasScale>(vec, params.a, vid);
    Lamport::clear_pos_zero(vec.data());
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_relaxed_16B(vec, push_ptrs[i], vid);
    }
  }

  // poll from local
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  vec_t pos_zero_vec;
  Lamport::fill_pos_zero(pos_zero_vec.data());

  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B(vec[i], poll_ptrs[i], vid);
      }
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        has_zero |= Lamport::has_pos_zero(vec[i].data());
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce_vec(vec);
    out_vec.store(params.output, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B(pos_zero_vec, poll_ptrs[i], vid);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  epoch.flip();
}
template<bool HasScale>
__global__ void finalize_push(const __grid_constant__ FinalizeParams params) {
 transport_body<true,HasScale>(params);
}
template<bool Fused,bool HasScale>
__global__ void virtual_eight_rank(const __grid_constant__ ProtoArgs p) {
 const uint32_t rank=blockIdx.y;
 FinalizeParams q{};
 q.a=p.ranks[rank];q.output=q.a.out;q.rank=rank;q.num_vecs=q.a.rows*q.a.hidden/8;
 for(int r=0;r<8;r++)q.ws.workspaces[r]=p.workspaces[r];
 q.ws.counter=p.counters[rank];q.ws.slot_bytes=p.slot_bytes;
 transport_body<Fused,HasScale>(q);
}
template __global__ void finalize_push<false>(const __grid_constant__ FinalizeParams);
template __global__ void finalize_push<true>(const __grid_constant__ FinalizeParams);
extern "C" size_t proto_args_size(){return sizeof(ProtoArgs);}
extern "C" int run_protocol(int fused,int scale,ProtoArgs* p,cudaStream_t stream) {
 int dev=0;cudaGetDevice(&dev);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,dev);
 if(p->controllers<1 || 8*p->controllers>prop.multiProcessorCount)return -1;
 for(int r=0;r<8;r++)if(p->ranks[r].hidden%256 || p->ranks[r].rows<2 || p->ranks[r].rows>8 || p->ranks[r].rows*p->ranks[r].hidden*2>p->slot_bytes)return -2;
 const dim3 grid(p->controllers,8),block(32);
 if(fused && scale)virtual_eight_rank<true,true><<<grid,block,0,stream>>>(*p);
 else if(fused)virtual_eight_rank<true,false><<<grid,block,0,stream>>>(*p);
 else virtual_eight_rank<false,false><<<grid,block,0,stream>>>(*p);
 return int(cudaGetLastError());
}
}
