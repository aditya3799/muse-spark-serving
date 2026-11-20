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
