#pragma once
#include <humming/utils/all.cuh>
template<class Base>
struct ResearchVirtualKContext : Base {
  using Base::Base;
  using BlockShape = typename Base::BlockShape;
  using WarpShape = typename Base::WarpShape;
  static constexpr uint32_t kOriginalKPartitions = BlockShape::K / WarpShape::K;
  static constexpr uint32_t K_WARPS = 1;
  static constexpr uint32_t kWarpIters = Base::kWarpIters * kOriginalKPartitions;
  CUDA_INLINE uint32_t k_warp_id() { return 0; }
  CUDA_INLINE uint32_t k_warp_offset() { return 0; }
};
