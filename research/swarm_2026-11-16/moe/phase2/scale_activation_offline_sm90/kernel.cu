#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe/phase2/scale_activation.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (ScaledActivationKernel<bf16_t, true>::run));
}  // namespace sglang
