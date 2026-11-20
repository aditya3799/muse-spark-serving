// Isolated diagnostic prototype. Exact pinned 2shot pull load/reduce and barriers.
// Output contains only this rank's contiguous complete token rows.
#pragma once
#include "distributed/custom_all_reduce.cuh"
namespace sglang {
template <typename Impl, typename T, uint32_t kWorldSize, bool kUsePDL>
ALL_REDUCE_KERNEL void muse_reduce_scatter_rows_kernel(const __grid_constant__ AllReducePullParams<kWorldSize> params) {
  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_total_vecs = params.num_vecs;
  const auto avg_vecs = num_total_vecs / kWorldSize;
  const auto rem_vecs = num_total_vecs % kWorldSize;
  const auto num_vecs = avg_vecs + (params.rank < rem_vecs ? 1 : 0);
  const auto vec_offset = params.rank * avg_vecs + min(params.rank, rem_vecs);
  const auto impl = Impl{params, vec_offset};
  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>{
      params.ws.semaphores.data(),
      params.rank,
      /*num_arrives=*/2,
  };
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    vec.store(params.output, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}

template <typename T, uint32_t kWorldSize, bool kUsePDL>
struct MuseReduceScatterRows {
 static void run(host::distributed::CommunicatorRef comm_ref,
                 tvm::ffi::TensorView in, tvm::ffi::TensorView out,
                 tvm::ffi::Optional<tvm::ffi::TensorView> graph_params_opt,
                 bool use_multicast) {
  using namespace host;
  SymbolicSize M{"tokens"}, K{"width"}, L{"local_tokens"};
  SymbolicDevice dev; dev.set_options<kDLCUDA>();
  TensorMatcher({M,K}).with_dtype<T>().with_device(dev).verify(in);
  TensorMatcher({L,K}).with_dtype<T>().with_device(dev).verify(out);
  CHECK_HOST(M.unwrap()>0 && M.unwrap()==L.unwrap()*kWorldSize);
  CHECK_HOST(in.IsContiguous() && out.IsContiguous());
  CHECK_HOST(in.data_ptr()!=out.data_ptr());
  CHECK_HOST(reinterpret_cast<uintptr_t>(in.data_ptr())%16==0);
  CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr())%16==0);
  const int64_t nbytes=in.numel()*sizeof(T);
  CHECK_HOST(nbytes%(16*kWorldSize)==0 && K.unwrap()*sizeof(T)%16==0);
  const uint32_t num_vecs=nbytes/16;
  CHECK_HOST(static_cast<int64_t>(num_vecs)*16==nbytes);
  const auto& comm=*comm_ref.get();
  CHECK_HOST(comm.get_world_size()==kWorldSize);
  const auto& pull=comm.get_pull_obj();
  const bool use_graph=graph_params_opt.has_value();
  CHECK_HOST(!(use_multicast && use_graph));
  const auto graph_params=use_graph ? graph_params_opt.value().data_ptr() : nullptr;
  const AllReducePullParams<kWorldSize> params{
    .output=out.data_ptr(), .num_vecs=num_vecs, .rank=pull.rank,
    .graph_params=static_cast<void* const*>(graph_params),
    .ws=pull.get_workspace<kWorldSize>(use_graph ? 0 : nbytes),
  };
  CHECK_HOST(!use_multicast || params.ws.mc_workspace!=nullptr);
  const auto stream=LaunchKernel::resolve_device(in.device());
  if (!use_graph) CHECK_CUDA(cudaMemcpyAsync(params.ws.workspaces[pull.rank], in.data_ptr(), nbytes, cudaMemcpyDeviceToDevice, stream));
  using V=device::AlignedVector<packed_t<T>,16/(sizeof(T)*2)>;
  using LS=LoadStoreImpl<V,kWorldSize,false>;
  using LSG=LoadStoreImpl<V,kWorldSize,true>;
  using MC=MultiCastImpl<V,kWorldSize,false>;
  if (use_multicast) {
    LaunchKernel(comm.get_pull_multicast_blocks(),512u,stream).enable_pdl(kUsePDL)(muse_reduce_scatter_rows_kernel<MC,T,kWorldSize,kUsePDL>,params);
  } else {
    const auto kernel=use_graph ? muse_reduce_scatter_rows_kernel<LSG,T,kWorldSize,kUsePDL> : muse_reduce_scatter_rows_kernel<LS,T,kWorldSize,kUsePDL>;
    LaunchKernel(comm.get_pull_blocks(),choose_block_size(num_vecs/kWorldSize),stream).enable_pdl(kUsePDL)(kernel,params);
  }
 }
};
} // namespace sglang
