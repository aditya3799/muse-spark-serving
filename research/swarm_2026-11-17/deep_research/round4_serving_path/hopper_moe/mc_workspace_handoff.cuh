// RESEARCH ONLY. Must be admitted collectively before any graph/storage mutation.
#pragma once
#include "custom_all_reduce.cuh"
namespace sglang {
struct ResearchMCWorkspaceHandoff {
 static tvm::ffi::Tensor run(const CommunicatorRef comm_ref,
                            const tvm::ffi::Tensor in,
                            const tvm::ffi::Tensor workspace_out) {
  using namespace host;
  constexpr uint32_t kWorldSize=8;
  constexpr bool kUsePDL=false;
  using T=bf16_t;
  using V=device::AlignedVector<bf16x2_t,4>;
  const auto& comm=*comm_ref.get();
  CHECK_HOST(comm.get_world_size()==8);
  CHECK_HOST(in.IsContiguous() && workspace_out.IsContiguous());
  CHECK_HOST(is_type<T>(in.dtype()) && is_type<T>(workspace_out.dtype()));
  CHECK_HOST(in.device().device_type==kDLCUDA && in.device()==workspace_out.device());
  CHECK_HOST(in.ndim()==2 && in.shape()[1]==8192);
  const auto m=in.shape()[0];
  CHECK_HOST(m>=32 && m<=1024);
  CHECK_HOST(workspace_out.ndim()==2 && workspace_out.shape()[0]==m && workspace_out.shape()[1]==8192);
  const int64_t nbytes=in.numel()*sizeof(T);
  CHECK_HOST(nbytes>=512*1024 && nbytes<=16*1024*1024 && nbytes%16==0);
  CHECK_HOST(reinterpret_cast<intptr_t>(in.data_ptr())%16==0);
  const auto stream=LaunchKernel::resolve_device(in.device());
  const auto& pull=comm.get_pull_obj();
  const auto ws=pull.get_workspace<8>(nbytes);
  CHECK_HOST(ws.mc_workspace!=nullptr);
  CHECK_HOST(comm.get_pull_multicast_blocks()==16);
  const auto local_workspace=ws.workspaces[pull.rank];
  CHECK_HOST(workspace_out.data_ptr()==local_workspace);
  // Input remains a separate complete projection. Direct-to-plane GEMM is NOT enabled.
  const auto input_begin=reinterpret_cast<uintptr_t>(in.data_ptr());
  const auto plane_begin=reinterpret_cast<uintptr_t>(local_workspace);
  CHECK_HOST(input_begin+nbytes<=plane_begin || plane_begin+nbytes<=input_begin);
  int device=0,major=0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_HOST(device==in.device().device_id);
  CHECK_CUDA(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
  CHECK_HOST(major==9);
    const auto cuda_memcpy = [&](void* dst, const void* src) {
      if constexpr (SGL_ARCH_HOPPER_OR_GREATER) {  // PDL memcpy is faster
        // based on micro benchmark, only enable when batch size is small + aligned
        constexpr int64_t threshold_MB = SGL_ARCH_BLACKWELL_OR_GREATER ? 1024 : 8;
        if (nbytes % device::kMaxVecBytes == 0 && nbytes <= threshold_MB * 1024 * 1024) {
          const auto copy_kernel = memcpy_kernel<device::kMaxVecBytes>;
          const uint32_t num_copy_vecs = nbytes / device::kMaxVecBytes;
          const uint32_t num_copy_threads = 128u;
          const uint32_t num_copy_blocks = div_ceil(num_copy_vecs, num_copy_threads);
          LaunchKernel(num_copy_blocks, num_copy_threads, stream)
              .enable_pdl(kUsePDL)(copy_kernel, dst, src, num_copy_vecs);
          return;
        }
      }
      // safe fallback to cudaMemcpyAsync for large size or older architecture
      CHECK_CUDA(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    };

  cuda_memcpy(local_workspace,in.data_ptr());
  const auto params=AllReducePullParams<8>{
   .output=workspace_out.data_ptr(),.num_vecs=static_cast<uint32_t>(in.numel()/8),
   .rank=pull.rank,.graph_params=nullptr,.ws=ws};
  using MC=MultiCastImpl<V,8,false>;
  const auto kernel=all_reduce_2shot_pull_kernel<MC,bf16_t,8,false>;
  LaunchKernel(comm.get_pull_multicast_blocks(),512u,stream).enable_pdl(false)(kernel,params);
  // No copy back: caller must immediately consume this LOCAL ordinary-VA view
  // with the unchanged norm kernel on this same stream before any plane reuse.
  return workspace_out;
 }
};
} // namespace sglang
