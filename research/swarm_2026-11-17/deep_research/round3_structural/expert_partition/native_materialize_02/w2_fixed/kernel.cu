// Isolated research materializer. Four adjacent packed BF16 values per thread.
#include <cuda_bf16.h>
#include <cstdint>
#include <vector_types.h>
#define N_VALUE 4096u
#define K_VALUE 4096u
#define SOURCE_TRANSPOSE false
#define TARGET_TRANSPOSE false
#define FIXED_GRID_LOOP true
#define THREADS_VALUE 256

__device__ __forceinline__ float decode_scale(uint32_t code) {
  uint32_t exponent = (code >> 3) & 15u;
  uint32_t fraction = code & 7u;
  uint32_t bits = ((exponent + 120u) << 23) | (fraction << 20);
  if (exponent == 0u) bits = __float_as_uint(__fmul_rn((float)fraction, 0.001953125f));
  if (exponent == 15u && fraction == 7u) bits = 0x7fc00000u;
  return __uint_as_float(bits | ((code & 128u) << 24));
}
__device__ __forceinline__ float decode_quarter(uint32_t code) {
  uint32_t mag = code & 7u;
  uint32_t bits = mag >= 2u ? 0x3e000000u + (mag << 22) : (mag == 1u ? 0x3e000000u : 0u);
  return __uint_as_float(bits | ((code & 8u) << 28));
}
__device__ __forceinline__ void write_quad(const uint32_t *__restrict__ weight, const uint8_t *__restrict__ scales, uint2 *__restrict__ output, uint32_t quad) {
  const uint32_t base = quad * 4u;
  const uint32_t group = base / (N_VALUE * 16u);
  const uint32_t within = base % (N_VALUE * 16u);
  const uint32_t nb = within / 1024u;
  const uint32_t rem = within % 1024u;
  const uint32_t lane = rem / 32u;
  const uint32_t inner = rem % 32u;
  const uint32_t i3 = inner / 8u;
  const uint32_t a = (inner / 4u) & 1u;
  const uint32_t word_index = group * (N_VALUE * 2u) + nb * 128u + lane * 4u + i3;
  const uint32_t packed = weight[word_index];
  const uint32_t scale_index = group * N_VALUE + nb * 64u + (lane / 4u) * 8u + i3 * 2u + (TARGET_TRANSPOSE ? 0u : a);
  const float scale0 = decode_scale(scales[scale_index]);
  float scale1 = scale0;
  if constexpr (TARGET_TRANSPOSE) scale1 = decode_scale(scales[scale_index + 1u]);
  uint32_t bits[4];
  #pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    uint32_t nibble = SOURCE_TRANSPOSE == TARGET_TRANSPOSE ? 2u*a + j/2u + 4u*(j%2u) : 2u*(j/2u) + a + 4u*(j%2u);
    uint32_t code = (packed >> (4u*nibble)) & 15u;
    float scaled = __fmul_rn(decode_quarter(code), j < 2u ? scale0 : scale1);
    bits[j] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(scaled));
  }
  uint2 result; result.x = bits[0] | (bits[1] << 16); result.y = bits[2] | (bits[3] << 16);
  output[quad] = result;
}
extern "C" __global__ __launch_bounds__(THREADS_VALUE) void materialize_packed(const uint32_t *__restrict__ weight, const uint8_t *__restrict__ scales, uint2 *__restrict__ output) {
  constexpr uint32_t quads = (N_VALUE * K_VALUE) / 4u;
  uint32_t quad = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (FIXED_GRID_LOOP) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (; quad < quads; quad += stride) write_quad(weight, scales, output, quad);
  } else {
    if (quad < quads) write_quad(weight, scales, output, quad);
  }
}

