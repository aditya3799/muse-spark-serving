#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication/phase3/reduce_scatter_unpool/reduce_scatter_rows.cuh"
namespace sglang {
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_2, (MuseReduceScatterRows<bf16_t, 2, true>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_4, (MuseReduceScatterRows<bf16_t, 4, true>::run));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_8, (MuseReduceScatterRows<bf16_t, 8, true>::run));
}  // namespace sglang
