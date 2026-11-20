#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "h200_mc_gate.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(local_address, (research_local_pull_address));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(staged, (ResearchMCWorkspaceHandoff::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(direct, (ResearchMCReadyPlaneHandoff::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(control, (AllReduceKernel<bf16_t,8,false>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_small, (MuseSparkNormResidual<8192,256,false,true,1,false,false>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(norm_large, (MuseSparkNormResidual<8192,512,false,false,1,false,false>::run));
}
