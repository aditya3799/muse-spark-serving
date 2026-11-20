#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
#include <cuda_runtime.h>
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

#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Decode original E4M3 bytes to full-value BF16, preserving signed zero.
// NaN bytes are handled by the incumbent coefficient lookup, below.
__device__ __forceinline__ uint16_t register_scale(unsigned byte) {
  unsigned magnitude = byte & 127u;
  unsigned bits = (magnitude << 4) + 0x3c00u;
  if (magnitude < 8u) {
    bits = __bfloat16_as_ushort(__float2bfloat16_rn(float(magnitude) * 0x1p-9f));
  }
  return uint16_t(bits | ((byte & 128u) << 8));
}

__device__ __forceinline__ uint32_t register_scale_pair(unsigned bytes) {
  return unsigned(register_scale(bytes & 255u)) |
         (unsigned(register_scale((bytes >> 8) & 255u)) << 16);
}

__device__ __forceinline__ uint32_t register_mul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 value = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                               *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&value);
}

__device__ __forceinline__ uint32_t register_normalize(uint32_t value) {
  const uint32_t magnitude = value & 0x01c001c0u;
  const uint32_t zeros = __vcmpeq2(magnitude, 0u);
  const uint32_t halves = __vcmpeq2(magnitude, 0x00400040u);
  const uint32_t biased = value + 0x3f003f00u - (halves & 0x00400040u);
  return (biased & ~zeros) | (value & 0x80008000u);
}

template<int R>
__device__ __forceinline__ uint32_t register_tiny_pair(uint32_t word) {
  if constexpr (R == 0) return word & 0x81c081c0u;
  if constexpr (R == 1) return __funnelshift_r(word, word, 3) & 0x81c081c0u;
  if constexpr (R == 2) return __funnelshift_r(word, word, 6) & 0x81c081c0u;
  return ((word & 0x001b001bu) * 4228u) & 0x81c081c0u;
}

template<int R, int NORMALIZE>
__device__ __forceinline__ uint32_t register_pair(uint32_t word, uint32_t bytes,
                                                uint32_t scale_pair, const uint16_t* table) {
  const unsigned byte = (bytes >> ((R & 1) * 8)) & 255u;
  // Preserve the original table's NaN payload/sign behavior exactly.
  if ((byte & 127u) == 127u) return serving_pair<R>(word, bytes, table);
  uint32_t q = register_tiny_pair<R>(word);
  if constexpr (NORMALIZE == 1) q = register_normalize(q);
  else q = register_mul2(q, 0x7e807e80u); // q*2^-126 -> exact full FP4 value
  uint32_t sf = (R & 1) ? (scale_pair >> 16) : (scale_pair & 65535u);
  sf |= sf << 16;
  return register_mul2(q, sf);
}

template<int VARIANT>
__global__ void dequant_probe(const uint4* words, const uint16_t* scales,
                              const uint16_t* table, uint4* out, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  uint4 q = words[index];
  unsigned sf = scales[index];
  unsigned pair = VARIANT ? register_scale_pair(sf) : 0;
  unsigned a[4] = {q.x,q.y,q.z,q.w};
#pragma unroll
  for (int j=0;j<4;++j) {
    uint4 value;
    if constexpr (VARIANT == 0) {
      value = make_uint4(serving_pair<0>(a[j],sf,table),serving_pair<1>(a[j],sf,table),
                         serving_pair<2>(a[j],sf,table),serving_pair<3>(a[j],sf,table));
    } else {
      value = make_uint4(register_pair<0,VARIANT>(a[j],sf,pair,table),register_pair<1,VARIANT>(a[j],sf,pair,table),
                         register_pair<2,VARIANT>(a[j],sf,pair,table),register_pair<3,VARIANT>(a[j],sf,pair,table));
    }
    out[index*4+j] = value;
  }
}
extern "C" int dequant_run(const uint4* words, const uint16_t* scales,
                            const uint16_t* table, uint4* out, int count,
                            int variant, cudaStream_t stream) {
  if (count < 1 || variant < 0 || variant > 2) return int(cudaErrorInvalidValue);
  int grid=(count+255)/256;
  if (variant == 0) dequant_probe<0><<<grid,256,0,stream>>>(words,scales,table,out,count);
  if (variant == 1) dequant_probe<1><<<grid,256,0,stream>>>(words,scales,table,out,count);
  if (variant == 2) dequant_probe<2><<<grid,256,0,stream>>>(words,scales,table,out,count);
  return int(cudaGetLastError());
}
