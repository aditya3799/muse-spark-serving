#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/mnt/d/OpenKernelChallenge/serving/research_2026-11-18_rl_prefill/route_activation.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (RouteActivationKernel<bf16_t, true, 8>::run));
}  // namespace sglang
