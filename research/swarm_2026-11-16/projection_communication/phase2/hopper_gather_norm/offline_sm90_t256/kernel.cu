#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/mnt/d/OpenKernelChallenge/serving/sglang/python/sglang/kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, (MuseSparkAllGatherNormResidual<8192, 8, 256, 1, true, true, true, true, true>::run));
}  // namespace sglang
