"""Render a held MC workspace-handoff entry point using the stock MC kernel."""
import hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
rel='sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh'
src=(ROOT/rel).read_text();ledger=json.loads((P/'SOURCE_COLLECTIVE_AUDIT.json').read_text())
assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==ledger['source_hashes'][rel]
a=src.index('    const auto cuda_memcpy =');b=src.index('    const uint32_t num_blocks =',a)
copy=src[a:b]
code='''// RESEARCH ONLY. Must be admitted collectively before any graph/storage mutation.
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
'''+copy+'''
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
'''
(P/'mc_workspace_handoff.cuh').write_bytes(code.encode())
# Export actual unmodified stock MC device kernel and both native norm geometries.
# Device bodies copied verbatim except wrapper signature/compile-time constants.
def body(text,name):
 i=text.index(name);a=text.index('{',i);j=a+1;depth=1
 while depth:depth+=(text[j]=='{')-(text[j]=='}');j+=1
 return text[a+1:j-1]
mc_body=body(src,'all_reduce_2shot_pull_kernel(')
normsrc=(ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh').read_text()
normbody=body(normsrc,'muse_spark_norm_residual_kernel(')
cu='''#include "mc_workspace_handoff.cuh"
#include "norm_residual.cuh"
using namespace sglang;
extern "C" __global__ __launch_bounds__(1024,1) void stock_mc_reduce(const AllReducePullParams<8> params) {
 using T=bf16_t;constexpr uint32_t kWorldSize=8;constexpr bool kUsePDL=false;
 using V=device::AlignedVector<bf16x2_t,4>;using Impl=MultiCastImpl<V,8,false>;
'''+mc_body+'}\n'
for name,threads,prefetch in [('small',256,'true'),('large',512,'false')]:
 cu+='extern "C" __global__ __launch_bounds__('+str(threads)+') void stock_norm_'+name+'(const NormResidualParams params) {\n'
 cu+=' constexpr int64_t kHidden=8192;constexpr int kNumThreads='+str(threads)+';constexpr bool kUsePDL=false,kPrefetchWeights='+prefetch+',kPostNormIsInputScale=false,kHasRouter=true,kPostWeight=true,kGate=true;\n'+normbody+'}\n'
(P/'offline_mc_handoff.cu').write_bytes(cu.encode())
report=dict(status='SOURCE_PROTOTYPE_HELD_NO_NATIVE_RUNTIME',host_adapter_sha256=hashlib.sha256(code.encode()).hexdigest(),offline_source_sha256=hashlib.sha256(cu.encode()).hexdigest(),stock_mc_body_sha256=hashlib.sha256(mc_body.encode()).hexdigest(),stock_norm_body_sha256=hashlib.sha256(normbody.encode()).hexdigest(),copy_input_lambda_verbatim=True,kernels_unchanged=['all_reduce_2shot_pull_kernel<MultiCastImpl,bf16,8,false>','singleCTA Muse norm body'],removed=['final workspace to ARoutput memcpy','separate ARoutput allocation'],not_removed=['inputstagingcopy','collectivebarriers','MCread/reduction/store','norm arithmetic'],required_python_guards=['active selectedalgorithm exactly2shot_pull +multicast +use_graphFalse; nooverride','H200/SM90 TP8 BF16 H8192,32<=M<=1024 and currentworkspacecapacity','defaultJIT boundary,DP/SPdisabled, noquantcommunications','view currentcomm._symm_tensor localpullrange and validatepointer in C++','consume borrowedview on samecurrentstream immediately; noexternalretention','nextreuse of plane mustfollowlocalnormcompletion and the existingnextcollectivepeerbarrier','no otherconcurrent users ofsamecomm plane; lifetime tokenheld throughnormcall'],fallback='Reject before partialGEMM/AR dispatch; original attention path remains complete fallback',same_precision_limit='Retains identical MCprimitive; does not replace it with rankordered software reduction',source_urls=['https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red','https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MULTICAST.html'])
(P/'MC_WORKSPACE_HANDOFF_CONTRACT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

