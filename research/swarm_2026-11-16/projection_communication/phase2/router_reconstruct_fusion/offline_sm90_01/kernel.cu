#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication/phase2/router_reconstruct_fusion/integration_sources/python/sglang/kernels/jit/csrc/muse_spark/routing.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_small, (fused_muse_spark_routing<8, 8, 0, 1, true, true>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_large, (fused_muse_spark_routing<8, 8, 0, 4, true, true>));
}  // namespace sglang
