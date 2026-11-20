#include <cuda_bf16.h>
#include <cuda_runtime.h>
using ushort = unsigned short;
using uint = unsigned int;
using uchar = unsigned char;
using ulong = unsigned long;
#ifdef _MSC_VER
#define __attribute__(x)
#endif
#include <flashinfer/norm.cuh>

// Expose the release's actual native CUDA implementation used by its public
// RMSNorm API in CUDA-backend mode. All tensors are ordinary contiguous BF16.
extern "C" __declspec(dllexport) int run_bf16_rms(
    __nv_bfloat16* input, __nv_bfloat16* weight, __nv_bfloat16* out,
    int rows, int hidden, float eps, cudaStream_t stream) {
  return int(flashinfer::norm::RMSNorm(
      input, weight, out, rows, hidden, hidden, hidden, eps, false, stream));
}



