#include <cuda_runtime.h>
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

template<int R, int H>
__device__ __forceinline__ unsigned serving_code(unsigned word) {
  if constexpr (R == 3) {
    unsigned half = word >> (H * 16);
    return ((half >> 4) & 1) | ((half & 3) << 1) | (half & 8);
  }
  unsigned rotated = __funnelshift_r(word, word, (6 + 3 * R + 16 * H) & 31);
  return (rotated & 7) | ((rotated >> 6) & 8);
}

template<int R>
__device__ __forceinline__ unsigned serving_pair(unsigned word, unsigned scales,
                                                  const uint16_t* table) {
  const uint16_t* row = table + ((scales >> ((R & 1) * 8)) & 255) * 16;
  return unsigned(row[serving_code<R,0>(word)]) |
         (unsigned(row[serving_code<R,1>(word)]) << 16);
}

__device__ __forceinline__ float serving_scaled(float dot, float scale) {
  float rounded = __bfloat162float(__float2bfloat16_rn(dot));
  return __bfloat162float(__float2bfloat16_rn(rounded * scale));
}

__device__ __forceinline__ __nv_bfloat16 serving_activation(float gate, float up, float scale) {
  float g = serving_scaled(gate, scale);
  float u = serving_scaled(up, scale);
  return __float2bfloat16_rn((g / (1.f + expf(-g))) * u);
}

__global__ void check(const unsigned* q, const unsigned* s, const uint16_t* lut,
                      unsigned* y, const float* accum, const float* scales,
                      __nv_bfloat16* activation, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  y[4*i] = serving_pair<0>(q[i], s[i], lut);
  y[4*i+1] = serving_pair<1>(q[i], s[i], lut);
  y[4*i+2] = serving_pair<2>(q[i], s[i], lut);
  y[4*i+3] = serving_pair<3>(q[i], s[i], lut);
  const int row = i/4096, col = i%4096;
  activation[i] = serving_activation(accum[row*8192+col], accum[row*8192+4096+col],scales[row]);
}
extern "C" int run(const unsigned* q, const unsigned* s, const uint16_t* lut, unsigned* y,
                   const float* a, const float* scales, __nv_bfloat16* act, int n, cudaStream_t stream) {
  check<<<(n+255)/256,256,0,stream>>>(q,s,lut,y,a,scales,act,n);
  return cudaGetLastError();
}
