

#define HUMMING_SHAPE_N 8192
#define HUMMING_SHAPE_K 4096
#define HUMMING_PAD_SHAPE_N 0
#define HUMMING_PAD_SHAPE_K 0
#define HUMMING_NUM_EXPERTS 32
#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 16
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE_N 0
#define HUMMING_USE_INT_WEIGHT_SCALE 0
#define HUMMING_USE_FUSED_E8M0_SCALE 0
#define HUMMING_HAS_ZERO_POINT 0
#define HUMMING_IS_FP_ZERO_POINT 0
#define HUMMING_HAS_BIAS 0
#define HUMMING_USE_PACKED_K_LAYOUT 0
#define HUMMING_MMA_TYPE_ID 1
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE 0
#define HUMMING_IS_BLOCK_WEIGHT_SCALE 0
#define HUMMING_IS_GROUP_WEIGHT_SCALE 1
#define HUMMING_IS_TENSOR_WEIGHT_SCALE 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE_2 0
#define HUMMING_IS_TENSOR_WEIGHT_SCALE_2 0
#define HUMMING_HAS_CHANNEL_WEIGHT_SCALE 0
#define HUMMING_HAS_TENSOR_WEIGHT_SCALE 0
#define HUMMING_HAS_INPUT_SCALE 0

#define HUMMING_USE_F16_ACCUM 0
#define HUMMING_USE_BATCH_INVARIANT 0
#define HUMMING_USE_M_MAJOR_INPUT_SCALE 0
#define HUMMING_GEMM_TYPE_ID 1
#define HUMMING_IS_INDEXED_GEMM 1
#define HUMMING_IS_GROUPED_GEMM 0
#define HUMMING_IS_GROUPED_CONTIGUOUS_GEMM 0
#define HUMMING_IS_GROUPED_MASKED_GEMM 0

#define HUMMING_USE_STREAM_K 0
#define HUMMING_NUM_STAGES 4
#define HUMMING_NUM_CTAS_PER_SM 1
#define HUMMING_USE_WARP_SPEC 0
#define HUMMING_USE_MBARRIER 0
#define HUMMING_USE_CP_ASYNC 1
#define HUMMING_USE_TMA 0
#define HUMMING_USE_TMA_A 0
#define HUMMING_USE_TMA_AS 0
#define HUMMING_USE_TMA_B 0
#define HUMMING_USE_TMA_C 0
#define HUMMING_USE_TMA_BS 0
#define HUMMING_USE_TMA_BS2 0
#define HUMMING_USE_TMA_BZP 0
#define HUMMING_USE_TMA_BIAS 0
#define HUMMING_REDUCE_OVERLAP_LAST_STAGE_ONLY 0
#define HUMMING_NUM_WRITE_SPLITS 1
#define HUMMING_MULTI_CAST_SIZE_A 1
#define HUMMING_MULTI_CAST_SIZE_B 1
#define HUMMING_RASTER_GROUP_M 1
#define HUMMING_NUM_THREADS 512
#define HUMMING_NUM_MATH_THREADS 512
#define HUMMING_NUM_LOAD_THREADS 512

#if 0
#include <humming/kernel/humming_ws.cuh>
#else
#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#pragma once

#include <humming/arith/exp_offset.cuh>
#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dequant.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class Ctx>
class MainloopArithmetic : F16Conversion<typename Ctx::ElementC> {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using ElementC = typename Ctx::ElementC;
  using ElementBS = typename Ctx::ElementBS;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kIsF16Accum = MmaOpClass::kCTypeBits == 16;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseIntWeightScale = Ctx::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kInputScaleGroupSize = kIsGroupInputScale ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = kIsGroupOrBlockWeightScale ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kNumZPGroupsPerMma = kUseMxmma && ElementA::kBits == 4 && kIsGroupWeightScale ? kPartMmaShapeK / kWeightScaleGroupSize : 1;

  static constexpr bool kUsePackedKLayout = Ctx::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = Ctx::kPackedKFactor;
  static constexpr uint32_t kNumKSlabs = WarpShape::K / kPartMmaShapeK;
  static constexpr uint2 kExpOffset = get_mainloop_exp_offset<
      ElementA, ElementB, ElementBS, kHasZeroPoint,
      kIsF16Accum, kIsGroupInputScale,
      kIsGroupOrBlockWeightScale && !kUseFusedE8m0Scale,
      MmaOpClass::kNativeMixed>();

  static constexpr uint32_t kDequantBSBits = (ElementA::kBits < 16 && !kIsF16Accum) ? 32 : 16;
  static constexpr uint32_t kNumSubBlocksM = CEIL_DIV(WarpShape::M, 16);
  static constexpr uint32_t kNumSubBlocksN = WarpShape::N / 16;
  static constexpr uint32_t kNumASPerSubBlock = kUseWgmma ? 4 : 2;
  static constexpr uint32_t kNumBSPerSubBlock = !kUseFusedE8m0Scale && !kUseWgmma && ElementA::kBits < 16 ? 4 : 2;
  static constexpr uint32_t kNumASPerGroup = kNumSubBlocksM * kNumASPerSubBlock;
  static constexpr uint32_t kNumBSPerGroup = kNumSubBlocksN * kNumBSPerSubBlock;

public:
  uint32_t as[2][kNumASPerGroup];
  uint32_t q_as[kNumASPerGroup];
  uint32_t bs[2][MAX(kNumBSPerGroup, 8) * ElementBS::kBits / 32];
  uint32_t dq_bs[MAX(kNumBSPerGroup, 8) * kDequantBSBits / 32];
  uint32_t zp[2][(kIsFpZeroPoint ? 4 : CEIL_DIV(ElementB::kBits, 4)) * kNumZPGroupsPerMma];

  uint32_t _dummy;

  CUDA_INLINE
  uint4 prepare_zp_for_dequant(uint32_t buffer_id, uint32_t index) {
    uint32_t zp_vals[4] = {0, 0, 0, 0};
    constexpr uint32_t kNumZPBits = 4 * CEIL_DIV(ElementB::kBits, 4);
    constexpr uint32_t kNumZPsPerInt = 32 / kNumZPBits;

    buffer_id = kIsChannelWeightScale ? 0 : buffer_id;

    if constexpr (kHasZeroPoint && !kIsFpZeroPoint) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) {
          uint32_t dzp_val;
          if constexpr (kNumZPGroupsPerMma > 1) {
            constexpr uint32_t kValuesPerLane = 32 / ElementA::kBits;
            constexpr uint32_t kZPBytesPerGroup = CEIL_DIV(kNumZPBits, 8);
            uint32_t k_offset = i * (kPartMmaShapeK / 2) + threadIdx.x % 4 * kValuesPerLane;
            uint32_t zp_group = k_offset / kWeightScaleGroupSize;
            uint32_t byte_id = index * kNumZPGroupsPerMma * kZPBytesPerGroup + zp_group * kZPBytesPerGroup;
            dzp_val = zp[buffer_id][byte_id / 4] >> (byte_id % 4 * 8);
            dzp_val = dzp_val >> (j * kNumZPBits);
          } else {
            uint32_t n_id = 0;
            if (WarpShape::N == 64 && ElementB::kBits > 4 && index >= 2) n_id = 1;
            dzp_val = zp[buffer_id][n_id];
            dzp_val = dzp_val >> ((index * 2 + j) % kNumZPsPerInt * kNumZPBits);
          }
          uint32_t zp_idx = kUseWgmma ? (i * 2 + j) : (j * 2 + i);
          zp_vals[zp_idx] = dequant_single_zero_point<ElementB, ElementA>(dzp_val);
        }
      }
    }

    return *reinterpret_cast<uint4 *>(zp_vals);
  };

  CUDA_INLINE
  void may_process_bs_before_apply_on_b(uint32_t j, uint32_t buffer_id) {

    if (j == 0) {
      if constexpr (ElementA::kBits == 16 && kIsBlockWeightScale) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        float bs_float = reinterpret_cast<float *>(bs[buffer_id])[0];
        dq_bs_scalar2_ptr[0] = this->float2num2(bs_float);
      }
    }

    if (j % 2 == 0) {
      if constexpr (ElementA::kBits == 16 && ElementBS::kBits == 8 && kIsGroupWeightScale && !kUseFusedE8m0Scale) {
        dequant<ElementBS, ElementA>(bs[buffer_id], dq_bs, 0);

        scalar_t *dq_bs_scalar_ptr = reinterpret_cast<scalar_t *>(dq_bs);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) {
          scalar_t tmp = dq_bs_scalar_ptr[2 + 4 * j];
          dq_bs_scalar_ptr[2 + 4 * j] = dq_bs_scalar_ptr[1 + 4 * j];
          dq_bs_scalar_ptr[1 + 4 * j] = tmp;
        }

        const scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);

        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 4; j++) {
          dq_bs_scalar2_ptr[j] = __hmul2(dq_bs_scalar2_ptr[j], scale_factor);
        }
      };
    };
  };

  CUDA_INLINE
  void may_apply_bs_and_zp_on_b(uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    may_process_bs_before_apply_on_b(j, buffer_id);

    static_assert(std::is_same<ElementA, BFloat16>::value && std::is_same<ElementB, Float4E2M1>::value && std::is_same<ElementBS, Float8E4M3>::value && kWeightScaleGroupSize == 16 && kExpOffset.x == 124 && kExpOffset.y == 120, "Exact quarter lookup contract");
    if constexpr (false && ElementA::kBits == 16 && kExpOffset.x) {
      uint32_t exp_val = kExpOffset.x + ((1 << (ElementA::kExponentBits - 1)) - 1);
      uint32_t scale_factor_uint = 0x00010001 * (exp_val << ElementA::kMantissaBits);
      scalar_t2 scale_factor = *reinterpret_cast<scalar_t2 *>(&scale_factor_uint);
      scalar_t2 *b_f16_ptr = reinterpret_cast<scalar_t2 *>(regs_b);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        b_f16_ptr[i] = __hmul2(b_f16_ptr[i], scale_factor);
      }
    }

    // apply bs and/or zp only when we use fp16/bf16 activation.
    if constexpr (ElementA::kBits == 16 && (kIsGroupWeightScale || kIsBlockWeightScale || kIsFpZeroPoint)) {

      scalar_t2 *b_f16_ptr = reinterpret_cast<scalar_t2 *>(regs_b);
      scalar_t2 bs_f16_ptr[2];
      scalar_t2 bzp_f16_ptr[2];

      if constexpr (kIsGroupWeightScale) {
        scalar_t *bs_half_ptr = reinterpret_cast<scalar_t *>(ElementBS::kBits == 8 ? &dq_bs[j] : &bs[buffer_id][j]);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 2; i++) {
          bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);
        }
      }

      if constexpr (kIsFpZeroPoint) {
        static_assert(kHasZeroPoint);
        static_assert(!ElementB::kIsSigned && ElementB::kIsIntegerType);
        uint32_t zp_buffer_id = kIsChannelWeightScale ? 0 : buffer_id;
        scalar_t2 bzp2_ptr = *reinterpret_cast<scalar_t2 *>(&zp[zp_buffer_id][j]);

        if constexpr (std::is_same<ElementA, BFloat16>::value && ElementB::kNumBits >= 7) {
          uint32_t exp_val = ((int32_t)kExpOffset.x - 133) + ((1 << (ElementA::kExponentBits - 1)) - 1);
          uint32_t scale_factor_uint = 0x00010001 * (exp_val << ElementA::kMantissaBits);
          scalar_t2 scale_factor = *reinterpret_cast<scalar_t2 *>(&scale_factor_uint);
          bzp2_ptr = __hmul2(bzp2_ptr, scale_factor);
        }

        scalar_t *bzp_ptr = reinterpret_cast<scalar_t *>(&bzp2_ptr);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 2; i++) {
          bzp_f16_ptr[i] = this->num2num2(bzp_ptr[i]);
        }
      }

      // Each warp apply bs or/and zp on a 16x16 weight block per time
      // The thread_id 0 process the following weight index of each block
      // [(0, 1), (0, 2)], [(0, 8), (0, 9)]
      // [(8, 1), (8, 2)], [(8, 8), (8, 9)]
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < 2; k++) {
          if constexpr (kIsFpZeroPoint) {
            scalar_t2 bzp_single = bzp_f16_ptr[kUseWgmma ? k : i];
            b_f16_ptr[i * 2 + k] = __hsub2(b_f16_ptr[i * 2 + k], bzp_single);
          }
          if constexpr (kIsGroupWeightScale) {
            scalar_t2 bs_single = bs_f16_ptr[kUseWgmma ? k : i];
            b_f16_ptr[i * 2 + k] = __hmul2(b_f16_ptr[i * 2 + k], bs_single);
          }
          if constexpr (kIsBlockWeightScale) {
            scalar_t2 bs_single = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
            b_f16_ptr[i * 2 + k] = __hmul2(b_f16_ptr[i * 2 + k], bs_single);
          }
        };
      };
    };
  };

  CUDA_INLINE
  void may_process_as_and_bs_before_apply_on_c(uint32_t m, uint32_t n, uint32_t k, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index, is_last_iter;
    if constexpr (kUsePackedKLayout) {
      k_index = (k + 1) * MmaShape::K;
      is_last_iter = (k + 1 == kNumKSlabs);
    } else {
      k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
      is_last_iter = iter_id == (Ctx::kWarpIters - 1);
    }
    uint32_t is_as_group_end = kIsGroupInputScale && k_index % kInputScaleGroupSize == 0;
    constexpr bool kProcessGroupWeightScale = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kProcessBlockWeightScale = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    uint32_t is_bs_group_end = (kProcessGroupWeightScale || kProcessBlockWeightScale) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kIsGroupInputScale && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kProcessGroupWeightScale || kProcessBlockWeightScale) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    if (m == 0 && n == 0 && should_apply_as) {
      if constexpr (kIsBlockWeightScale) {
        float *as_float_ptr = reinterpret_cast<float *>(as[buffer_id]);
        float *q_as_float_ptr = reinterpret_cast<float *>(q_as);
        float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < WarpShape::M / 8 * (kUseWgmma ? 2 : 1); i++) {
          as_float_ptr[i] = as_float_ptr[i] * block_bs_float;
        }
      }

      if constexpr (kIsF16Accum) {
        float2 *as_float2_ptr = reinterpret_cast<float2 *>(as[buffer_id]);
        float *as_float_ptr = reinterpret_cast<float *>(as[buffer_id]);
        scalar_t2 *as_scalar2_ptr = reinterpret_cast<scalar_t2 *>(as[buffer_id]);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
          if constexpr (kUseWgmma) {
            as_scalar2_ptr[i] = this->float22num2(as_float2_ptr[i]);
          } else {
            as_scalar2_ptr[i] = this->float2num2(as_float_ptr[i]);
          }

          if constexpr (kExpOffset.y && !kIsGroupWeightScale) {
            scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
            as_scalar2_ptr[i] = __hmul2(as_scalar2_ptr[i], scale_factor);
          }
        };
      };
    }

    if (m == 0 && n == 0 && should_apply_bs) {
      if constexpr (kUseIntWeightScale) {
        int16_t *bs_vals = reinterpret_cast<int16_t *>(bs[buffer_id]);
        int32_t *dq_bs_vals = reinterpret_cast<int32_t *>(dq_bs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup; i++)
          dq_bs_vals[i] = (int32_t)bs_vals[i];
      } else if constexpr (kIsF16Accum && ElementBS::kBits == 16) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        scalar_t2 *bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(bs[buffer_id]);

        if constexpr (kExpOffset.y) {
          scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
            dq_bs_scalar2_ptr[i] = __hmul2(bs_scalar2_ptr[i], scale_factor);
        }
      } else if constexpr (kIsF16Accum && kIsBlockWeightScale) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
        dq_bs_scalar2_ptr[0] = this->float2num2(reinterpret_cast<float *>(bs[buffer_id])[0]);
        dq_bs_scalar2_ptr[0] = __hmul2(dq_bs_scalar2_ptr[0], scale_factor);
      } else if constexpr (kIsF16Accum && ElementBS::kBits == 8 && kIsGroupWeightScale) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumBSPerGroup, 8); i++) {
          dequant<ElementBS, ElementC>(bs[buffer_id], dq_bs, i);

          scalar_t *dq_bs_scalar_ptr = reinterpret_cast<scalar_t *>(dq_bs);
          PRAGMA_UNROLL
          for (uint32_t j = 0; j < 2; j++) {
            scalar_t tmp = dq_bs_scalar_ptr[2 + 4 * j];
            dq_bs_scalar_ptr[2 + 4 * j] = dq_bs_scalar_ptr[1 + 4 * j];
            dq_bs_scalar_ptr[1 + 4 * j] = tmp;
          }
        }

        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        constexpr uint32_t exp_offset = get_dtype_dequant_exp_offset<ElementC, ElementBS>();
        scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, exp_offset>();
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
          dq_bs_scalar2_ptr[i] = __hmul2(dq_bs_scalar2_ptr[i], scale_factor);

        if constexpr (kExpOffset.y) {
          scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
            dq_bs_scalar2_ptr[i] = __hmul2(dq_bs_scalar2_ptr[i], scale_factor);
        }

      } else if constexpr (!kIsF16Accum && ElementBS::kBits == 8) {
        using F8x4 = typename F8Conversion<ElementBS>::scalar_t4;
        F8x4 *bs_vals = reinterpret_cast<F8x4 *>(bs[buffer_id]);
        float4 *dq_bs_vals = reinterpret_cast<float4 *>(dq_bs);

        constexpr uint32_t kFullPackets = kNumBSPerGroup / 4;
        constexpr uint32_t kTailScales = kNumBSPerGroup % 4;

        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kFullPackets; i++)
          dq_bs_vals[i] = F8Conversion<ElementBS>::num42float4(bs_vals[i]);

        if constexpr (kTailScales != 0) {
          uint32_t packed = 0;
          const uint8_t *src = reinterpret_cast<const uint8_t *>(bs[buffer_id]);
          uint8_t *dst = reinterpret_cast<uint8_t *>(&packed);

          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kTailScales; i++)
            dst[i] = src[kFullPackets * 4 + i];

          dq_bs_vals[kFullPackets] =
              F8Conversion<ElementBS>::num42float4(*reinterpret_cast<F8x4 *>(&packed));
        }
      } else if constexpr (!kIsF16Accum && ElementBS::kBits == 16) {
        using F16x2 = typename F16Conversion<ElementBS>::scalar_t2;
        F16x2 *bs_vals = reinterpret_cast<F16x2 *>(bs[buffer_id]);
        float2 *dq_bs_vals = reinterpret_cast<float2 *>(dq_bs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
          dq_bs_vals[i] = F16Conversion<ElementBS>::num22float2(bs_vals[i]);
      }
    }
  };

  CUDA_INLINE
  void may_apply_as_and_bs_on_mma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t n, uint32_t k, uint32_t iter_id) {
    if constexpr (ElementA::kBits == 16) return;
    if constexpr (!kIsGroupInputScale && !kIsGroupWeightScale && !kIsBlockWeightScale) return;
    if constexpr (kUseWgmma) return;
    constexpr bool kApplyGroupWeightScaleOnC = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyBlockWeightScaleOnC = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyGroupInputScaleOnC = kIsGroupInputScale;
    if constexpr (kUseFusedE8m0Scale && !kApplyGroupInputScaleOnC) return;

    may_process_as_and_bs_before_apply_on_c(m, n, k, iter_id);

    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
    uint32_t is_last_iter = iter_id == (Ctx::kWarpIters - 1);
    uint32_t is_as_group_end = kApplyGroupInputScaleOnC && k_index % kInputScaleGroupSize == 0;
    uint32_t is_bs_group_end = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kApplyGroupInputScaleOnC && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    using CRegistersArrayType = typename MmaOpClass::CRegisters[2][WarpShape::M / MmaShape::M][WarpShape::N / MmaShape::N];
    auto &regs_c = *reinterpret_cast<CRegistersArrayType *>(regs_c_ptr);

    PRAGMA_UNROLL
    for (uint32_t index = 0; index < MmaShape::M * MmaShape::N / 64; index++) {
      uint32_t inner_m = index % (MmaShape::M / 8);
      uint32_t inner_n = index / (MmaShape::M / 8);

      if constexpr (kIsF16Accum) {
        scalar_t2 &part_regs_c0 = reinterpret_cast<scalar_t2 *>(regs_c[0][m][n])[index];
        scalar_t2 &part_regs_c1 = reinterpret_cast<scalar_t2 *>(regs_c[1][m][n])[index];

        scalar_t2 as_val = reinterpret_cast<scalar_t2 *>(as[buffer_id])[m * MmaShape::M / 8 + inner_m];
        scalar_t2 bs_val;
        if constexpr (!kIsGroupWeightScale && kIsBlockWeightScale) {
          bs_val = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
        } else if constexpr (ElementBS::kBits == 8 || kExpOffset.y) {
          bs_val = reinterpret_cast<scalar_t2 *>(dq_bs)[n * MmaShape::N / 8 + inner_n];
        } else {
          bs_val = reinterpret_cast<scalar_t2 *>(bs[buffer_id])[n * MmaShape::N / 8 + inner_n];
        }

        if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, __hmul2(as_val, bs_val), part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, as_val, part_regs_c1);
        }
        reinterpret_cast<uint32_t *>(&part_regs_c0)[0] = 0;

      } else {
        int2 &part_int_regs_c0 = reinterpret_cast<int2 *>(regs_c[0][m][n])[index];
        int2 &part_int_regs_c1 = reinterpret_cast<int2 *>(regs_c[1][m][n])[index];
        float2 &part_regs_c0 = reinterpret_cast<float2 *>(regs_c[0][m][n])[index];
        float2 &part_regs_c1 = reinterpret_cast<float2 *>(regs_c[1][m][n])[index];
        if constexpr (kUseIntWeightScale) {
          static_assert(std::is_same<ValTypeC, int32_t>::value);
          static_assert(!kIsGroupInputScale);
          static_assert(kIsGroupWeightScale);

          int2 &bs_vals = reinterpret_cast<int2 *>(dq_bs)[n * MmaShape::N / 8 + inner_n];

          part_int_regs_c1.x += bs_vals.x * part_int_regs_c0.x;
          part_int_regs_c1.y += bs_vals.y * part_int_regs_c0.y;

          part_int_regs_c0.x = 0;
          part_int_regs_c0.y = 0;
        } else {
          if constexpr (std::is_same<ValTypeC, int32_t>::value) {
            part_regs_c0.x = __int2float_rn(part_int_regs_c0.x);
            part_regs_c0.y = __int2float_rn(part_int_regs_c0.y);
          }

          float &as_val = reinterpret_cast<float *>(as[buffer_id])[m * MmaShape::M / 8 + inner_m];
          float2 &bs_vals = reinterpret_cast<float2 *>(dq_bs)[n * MmaShape::N / 8 + inner_n];
          float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];

          if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_val * bs_vals.x * part_regs_c0.x;
            part_regs_c1.y += as_val * bs_vals.y * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += bs_vals.x * part_regs_c0.x;
            part_regs_c1.y += bs_vals.y * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
            part_regs_c1.x += block_bs_float * part_regs_c0.x;
            part_regs_c1.y += block_bs_float * part_regs_c0.y;
          } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_val * part_regs_c0.x;
            part_regs_c1.y += as_val * part_regs_c0.y;
          }

          part_regs_c0.x = 0;
          part_regs_c0.y = 0;
        }
      }
    }
  }

  CUDA_INLINE
  void may_apply_as_and_bs_on_wgmma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t k, uint32_t iter_id, uint32_t delta_m) {
    if constexpr (ElementA::kBits == 16) return;
    if constexpr (!kIsGroupInputScale && !kIsGroupWeightScale && !kIsBlockWeightScale) return;
    if constexpr (!kUseWgmma) return;
    constexpr bool kApplyGroupWeightScaleOnC = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyBlockWeightScaleOnC = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyGroupInputScaleOnC = kIsGroupInputScale;
    if constexpr (kUseFusedE8m0Scale && !kApplyGroupInputScaleOnC) return;

    may_process_as_and_bs_before_apply_on_c(m, 0, k, iter_id);

    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index, is_last_iter;
    if constexpr (kUsePackedKLayout) {
      k_index = (k + 1) * MmaShape::K;
      is_last_iter = (k + 1 == kNumKSlabs);
    } else {
      k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
      is_last_iter = iter_id == (Ctx::kWarpIters - 1);
    }
    uint32_t is_as_group_end = kApplyGroupInputScaleOnC && k_index % kInputScaleGroupSize == 0;
    uint32_t is_bs_group_end = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kApplyGroupInputScaleOnC && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    using CRegistersArrayType = typename MmaOpClass::CRegisters[2][WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];
    auto &regs_c = *reinterpret_cast<CRegistersArrayType *>(regs_c_ptr);

    PRAGMA_UNROLL
    for (uint32_t index = 0; index < MmaShape::M * 16 / 64; index++) {
      uint32_t inner_n = index / 2;
      uint32_t inner_m = index % 2;

      if constexpr (kIsF16Accum) {
        scalar_t2 &part_regs_c0 = reinterpret_cast<scalar_t2 *>(regs_c[0][m][0])[index];
        scalar_t2 &part_regs_c1 = reinterpret_cast<scalar_t2 *>(regs_c[1][delta_m + m][0])[index];

        scalar_t2 as_val = reinterpret_cast<scalar_t2 *>(as[buffer_id])[inner_n];
        scalar_t2 bs_val;
        if constexpr (!kIsGroupWeightScale && kIsBlockWeightScale) {
          bs_val = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
        } else if constexpr (ElementBS::kBits == 8 || kExpOffset.y) {
          bs_val = this->num2num2(reinterpret_cast<scalar_t *>(dq_bs)[(delta_m + m) * 2 + inner_m]);
        } else {
          bs_val = this->num2num2(reinterpret_cast<scalar_t *>(bs[buffer_id])[(delta_m + m) * 2 + inner_m]);
        }

        float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];

        if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, __hmul2(as_val, bs_val), part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, as_val, part_regs_c1);
        }
      } else {
        int2 &part_int_regs_c0 = reinterpret_cast<int2 *>(regs_c[0][m][0])[index];
        int2 &part_int_regs_c1 = reinterpret_cast<int2 *>(regs_c[1][delta_m + m][0])[index];
        float2 &part_regs_c0 = reinterpret_cast<float2 *>(regs_c[0][m][0])[index];
        float2 &part_regs_c1 = reinterpret_cast<float2 *>(regs_c[1][delta_m + m][0])[index];

        if constexpr (kUseIntWeightScale) {
          static_assert(std::is_same<ValTypeC, int32_t>::value);
          static_assert(!kIsGroupInputScale);
          static_assert(kIsGroupWeightScale);

          int32_t bs_val = reinterpret_cast<int32_t *>(dq_bs)[(delta_m + m) * 2 + inner_m];
          part_int_regs_c1.x += bs_val * part_int_regs_c0.x;
          part_int_regs_c1.y += bs_val * part_int_regs_c0.y;
        } else {
          if constexpr (std::is_same<ValTypeC, int32_t>::value) {
            part_regs_c0.x = __int2float_rn(part_int_regs_c0.x);
            part_regs_c0.y = __int2float_rn(part_int_regs_c0.y);
          }

          float2 &as_vals = *reinterpret_cast<float2 *>(&as[buffer_id][inner_n * 2]);
          float &bs_val = reinterpret_cast<float *>(dq_bs)[(delta_m + m) * 2 + inner_m];
          float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];

          if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_vals.x * bs_val * part_regs_c0.x;
            part_regs_c1.y += as_vals.y * bs_val * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += bs_val * part_regs_c0.x;
            part_regs_c1.y += bs_val * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
            part_regs_c1.x += block_bs_float * part_regs_c0.x;
            part_regs_c1.y += block_bs_float * part_regs_c0.y;
          } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_vals.x * part_regs_c0.x;
            part_regs_c1.y += as_vals.y * part_regs_c0.y;
          }
        }
      }
    }
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_bs_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(bs[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_as_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(as[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_zp_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(zp[buffer_id]);
  };
};


#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#pragma once

#include <humming/mma/mxmma.cuh>
#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t make_wgmma_smem_desc(uint32_t addr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};



CUDA_INLINE uint32_t research_fp4_quarter_pair(uint32_t val) {
  uint32_t select = ((val >> 12) & 7u) | ((val >> 20) & 0x700u);
  uint32_t lo, hi;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(0xc0800000u), "r"(0xc0804000u), "r"(select));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(0x3e3e3e00u), "r"(0x3f3f3f3fu), "r"(select));
  return (lo & 0x00ff00ffu) | ((hi & 0x00ff00ffu) << 8) | (val & 0x80008000u);
}

template <class Ctx, class ArithClass>
struct WGMMA {
public:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using MmaShape = typename Ctx::MmaShape;
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using CRegistersType = typename MmaOpClass::CRegisters;
  using CRegistersArrayType = CRegistersType[WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;

  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kSwizzleBytes = ElementA::kBits * BlockShape::K >= 1024 ? 128 : 64;
  static constexpr uint32_t kNumWarpShapeNSplits = WarpShape::N == ElementA::kBits * 2 ? 2 : 1;

  static constexpr bool kUsePackedKLayout = Ctx::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = Ctx::kPackedKFactor;
  static constexpr uint32_t kWarpIters = Ctx::kWarpIters;
  static constexpr uint32_t kNumKSlabs = WarpShape::K / kPartMmaShapeK;
  static constexpr uint32_t kRegsBKDim = kUsePackedKLayout ? kNumKSlabs : (kPartMmaShapeK / MmaShape::K);

  Ctx &ctx;
  ArithClass &arith;
  uint32_t regs_qb[2][ElementB::kBits * (16 / ElementA::kBits)];
  typename MmaOpClass::BRegisters regs_b[2][kUsePackedKLayout ? 1 : (WarpShape::N * 4 / MmaShape::N / kPackedKFactor)][kRegsBKDim];
  CRegistersArrayType regs_c[2];
  uint32_t smem_offset = 0;

  CUDA_INLINE
  WGMMA(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
    constexpr uint32_t kSwizzleSizeK = kSwizzleBytes * 8 / ElementA::kBits;
    static_assert(kSwizzleSizeK >= WarpShape::K);

    const uint32_t row_offset = ctx.m_warp_offset();
    const uint32_t col_offset = ctx.k_warp_offset();

    smem_offset = row_offset * (kSwizzleBytes / 16);
    smem_offset += (col_offset % kSwizzleSizeK) * ElementA::kBits / 128;
    smem_offset += (col_offset / kSwizzleSizeK) * (BlockShape::M * kSwizzleBytes / 16);
    smem_offset = smem_offset * sizeof(int4);
  }

  CUDA_INLINE
  void zero_accum() {
    uint32_t *regs_c_ptr = regs_c_as_ptr();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs_c) / 4; i++) {
      regs_c_ptr[i] = 0;
    };
  };

  CUDA_INLINE
  void transform_b(uint32_t buffer_id, uint32_t iter_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) return;

    if constexpr (kUseFusedE8m0Scale) {
      uint32_t *regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id]);
      fused_dequant_for_mxfp4<ElementA, WarpShape::N / 16, true>(regs_qb[buffer_id], regs_b_ptr, arith.bs[buffer_id]);
    } else {
      if constexpr (ElementB::kBits == 1 && kNumWarpShapeNSplits == 2) {
        regs_qb[buffer_id][0] = regs_qb[buffer_id][0] >> (ctx.warp_id() % 2 * (ElementA::kBits / 2));
      }

      constexpr uint32_t kTransformIters = kUsePackedKLayout ? kNumKSlabs : (WarpShape::N / (MmaShape::N / 4));
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kTransformIters; i++) {
        uint32_t *regs_b_ptr;
        if constexpr (kUsePackedKLayout) {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][0][i]);
        } else {
          regs_b_ptr = reinterpret_cast<uint32_t *>(regs_b[buffer_id][i / kPackedKFactor][i % kPackedKFactor]);
        }
        uint32_t zp_index = kUsePackedKLayout ? iter_id : i;
        uint4 zp_vals = arith.prepare_zp_for_dequant(buffer_id, zp_index);
        uint32_t *zp_vals_ptr = reinterpret_cast<uint32_t *>(&zp_vals);
        PRAGMA_UNROLL
        for (uint32_t pair = 0; pair < 4; ++pair) {
          uint32_t val = regs_qb[buffer_id][i] << (pair * 4);
          regs_b_ptr[3 - pair] = research_fp4_quarter_pair(val);
        }
        arith.may_apply_bs_and_zp_on_b(regs_b_ptr, i, buffer_id);
      };
    }
  };

  CUDA_INLINE
  void run(uint32_t stage_id, uint32_t iter_id) {
    static_assert(WarpShape::M == MmaShape::M);
    static_assert(kPartMmaShapeK == MmaShape::K);
    uint32_t buffer_id = iter_id % 2;

    const uint32_t smem_base = cast_smem_ptr_to_uint(&ctx.smem);
    constexpr uint32_t kItersPerHalf = kWarpIters / kPackedKFactor;
    constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
    constexpr uint32_t kRunKLoop = kUsePackedKLayout ? kNumKSlabs : kPackedKFactor;

    uint32_t delta_m = kUsePackedKLayout ? iter_id : 0;
    uint32_t delta_j = final_regs_c_index() == 0 ? delta_m : 0;

    wgmma_fence();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      uint32_t k_slab = kUsePackedKLayout ? k : ((iter_id % kItersPerHalf) * kPackedKFactor + k);
      uint32_t smem_addr = smem_base + offsetof(SharedStorage, stages) + stage_id * sizeof(typename SharedStorage::StageStorage);
      smem_addr += k_slab * 2 * sizeof(int4) + smem_offset;
      uint64_t desc = make_wgmma_smem_desc<kSwizzleBytes>(smem_addr);

      bool scale_d = true;
      if constexpr (ElementA::kBits != 16 && Ctx::kInputScaleGroupSize > 0) {
        scale_d = (k_slab * kPartMmaShapeK) % Ctx::kInputScaleGroupSize > 0;
      }
      if constexpr (!kUseFusedE8m0Scale && ElementA::kBits != 16 && Ctx::kWeightScaleGroupSize > 0) {
        scale_d = scale_d && (k_slab * kPartMmaShapeK) % Ctx::kWeightScaleGroupSize > 0;
      }

      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        MmaOpClass::fma(desc, regs_b[buffer_id][j][k], regs_c[0][delta_j + j][0], scale_d);
      }
    }

    wgmma_commit();
    wgmma_wait<0>();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id, delta_m);
      }
    }
  };

  CUDA_INLINE void may_fence_regs(uint32_t delta_j) {
    if constexpr (final_regs_c_index() != 0) {
      constexpr uint32_t kNumIters = kUsePackedKLayout ? 1 : (WarpShape::N / (MmaShape::N / 4) / kPackedKFactor);
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++)
        fence_regs(regs_c[0][delta_j + j][0]);
    }
  }

  template <class T>
  CUDA_INLINE void fence_regs(T &regs) {
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < sizeof(T) / 4; r++) {
      warpgroup_fence_operand(reinterpret_cast<uint32_t *>(regs)[r]);
    }
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_qb_as_ptr(uint32_t buffer_id) {
    if constexpr (std::is_same<ElementA, ElementB>::value) {
      return reinterpret_cast<T *>(regs_b[buffer_id]);
    } else {
      return reinterpret_cast<T *>(regs_qb[buffer_id]);
    };
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_c_as_ptr(uint32_t buffer_id = 0) {
    return reinterpret_cast<T *>(regs_c[buffer_id]);
  };

  static constexpr uint32_t final_regs_c_index() {
    if constexpr (ElementA::kBits < 16 && Ctx::kInputScaleGroupSize > 0) return 1;
    if constexpr (ElementA::kBits < 16 && !kUseFusedE8m0Scale && (Ctx::kIsGroupWeightScale || Ctx::kIsBlockWeightScale)) return 1;
    return 0;
  };

  template <class T = uint32_t>
  CUDA_INLINE T *final_regs_c_as_ptr() {
    return regs_c_as_ptr<T>(final_regs_c_index());
  };
};

#include <humming/mma/wmma.cuh>


template <MmaType kMmaType, class Ctx, class ArithClass>
struct MmaSelector;

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MMA, Ctx, ArithClass> {
  using Type = WMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::WGMMA, Ctx, ArithClass> {
  using Type = WGMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MXMMA, Ctx, ArithClass> {
  using Type = MXMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
using Mma = typename MmaSelector<Ctx::kMmaType, Ctx, ArithClass>::Type;


#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};

CUDA_INLINE const void *param_to_ptr(const CUtensorMap &x) { return &x; }
CUDA_INLINE const void *param_to_ptr(void *const &x) { return x; }


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS>::Type AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS2>::Type BS2,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  uint64_t debug_start_clock = debug_kernel_timer_start();
  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Ctx = KernelContext<
      MmaOpClass, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<Ctx>;
  using ProducerPipeline = ProducerPipeline<Ctx>;
  using ConsumerPipeline = ConsumerPipeline<Ctx>;
  using MainloopArithmetic = MainloopArithmetic<Ctx>;
  using EpilogueArithmetic = EpilogueArithmetic<Ctx>;
  using MMA = Mma<Ctx, MainloopArithmetic>;
  using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;
  using S2RMemoryPipeline = S2RMemoryPipeline<Ctx, MMA, Epilogue>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  const KernelParams params{
      shape_m, top_k, use_int64_expert_layout,
      param_to_ptr(A), param_to_ptr(B), param_to_ptr(AS), param_to_ptr(BS),
      param_to_ptr(BZP), param_to_ptr(Bias), param_to_ptr(C), param_to_ptr(BS2),
      sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr,
      tensor_map_buffer, locks};
  auto ctx = Ctx(smem, params);

  auto scheduler = Scheduler(ctx);
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(ctx, mainloop_arith);
  auto epilogue = Epilogue(ctx, epilogue_arith);
  auto producer = ProducerPipeline(ctx);
  auto consumer = ConsumerPipeline(ctx);
  auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

  producer.init_mbarrier();
  mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();

  while (scheduler.get_next_block()) {
    debug_kernel_timeout_check(debug_start_clock);
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (TuningConfig::kUseTmaC) {
      tma_wait_store_group<0, true>();
      __syncthreads();
    }
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    while (slice_iters) {
      debug_kernel_timeout_check(debug_start_clock);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        debug_kernel_timeout_check(debug_start_clock);
        if (slice_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
        slice_iters--;
        if (!slice_iters) break;
      };
    };

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB > 1) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};

#endif

class MmaOpClass {
public:
  static constexpr MmaType kMmaType = MmaType::WGMMA;
  using MmaShape = Shape<24, 64, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using BRegisters = uint32_t[4];
  using CRegisters = float[12];
  using DRegisters = float[12];

  CUDA_INLINE
  static void fma(uint64_t &desc, uint32_t *b, float *d, bool pred = true) {
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %17, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      :  "r"(b[0]),  "r"(b[1]),  "r"(b[2]),  "r"(b[3]),
         "l"(desc), "r"((uint32_t)pred)
    );
  };
};

class LayerConfig {
public:
  static constexpr auto kShapeN = 8192u;
  static constexpr auto kShapeK = 4096u;
  static constexpr auto kPadShapeN = 0u;
  static constexpr auto kPadShapeK = 0u;
  static constexpr auto kNumExperts = 32u;
  static constexpr auto kInputScaleGroupSize = 0u;
  static constexpr auto kWeightScaleGroupSize = 16u;
  static constexpr auto kWeightScaleGroupSizeN = 0u;
  static constexpr auto kWeightScaleType = WeightScaleType::GROUP;
  static constexpr auto kWeightScale2Type = WeightScale2Type::NONE;
  static constexpr auto kUseIntWeightScale = false;
  static constexpr auto kUseFusedE8m0Scale = false;
  static constexpr auto kHasZeroPoint = false;
  static constexpr auto kIsFpZeroPoint = false;
  static constexpr auto kHasBias = false;
  static constexpr auto kMmaType = MmaType::WGMMA;
  static constexpr auto kUsePackedKLayout = false;
  static constexpr auto kMmaTypeId = 1u;
  static constexpr auto kIsChannelWeightScale = false;
  static constexpr auto kIsBlockWeightScale = false;
  static constexpr auto kIsGroupWeightScale = true;
  static constexpr auto kIsTensorWeightScale = false;
  static constexpr auto kIsChannelWeightScale2 = false;
  static constexpr auto kIsTensorWeightScale2 = false;
  static constexpr auto kHasChannelWeightScale = false;
  static constexpr auto kHasTensorWeightScale = false;
  static constexpr auto kHasInputScale = false;
};

class ComputeConfig {
public:
  static constexpr auto kUseF16Accum = false;
  static constexpr auto kUseBatchInvariant = false;
  static constexpr auto kUseMMajorInputScale = false;
  static constexpr auto kGemmType = GemmType::INDEXED;
  static constexpr auto kGemmTypeId = 1u;
  static constexpr auto kIsIndexedGemm = true;
  static constexpr auto kIsGroupedGemm = false;
  static constexpr auto kIsGroupedContiguousGemm = false;
  static constexpr auto kIsGroupedMaskedGemm = false;
};

class TuningConfig {
public:
  static constexpr auto kUseStreamK = false;
  static constexpr auto kNumStages = 4u;
  static constexpr auto kNumCtasPerSm = 1u;
  static constexpr auto kUseWarpSpec = false;
  static constexpr auto kUseMBarrier = false;
  static constexpr auto kUseCpAsync = true;
  static constexpr auto kUseTma = false;
  static constexpr auto kUseTmaA = false;
  static constexpr auto kUseTmaAS = false;
  static constexpr auto kUseTmaB = false;
  static constexpr auto kUseTmaC = false;
  static constexpr auto kUseTmaBS = false;
  static constexpr auto kUseTmaBS2 = false;
  static constexpr auto kUseTmaBZP = false;
  static constexpr auto kUseTmaBias = false;
  static constexpr auto kReduceOverlapLastStageOnly = false;
  static constexpr auto kNumWriteSplits = 1u;
  static constexpr auto kMultiCastSizeA = 1u;
  static constexpr auto kMultiCastSizeB = 1u;
  static constexpr auto kRasterGroupM = 1u;
  static constexpr auto kNumThreads = 512u;
  static constexpr auto kNumMathThreads = 512u;
  static constexpr auto kNumLoadThreads = 512u;
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<24, 256, 128>,
    Shape<24, 32, 64>,
    FloatingPointType<16, 8, 7>,
    FloatingPointType<4, 2, 1>,
    FloatingPointType<8, 4, 3>,
    LayerConfig,
    ComputeConfig,
    TuningConfig>;



extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A = 
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeA * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_B = 
    SharedStorageType::kNumStages * SharedStorageType::kStageSizeB * sizeof(int4);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);

extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = 8192;
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = 4096;

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = 24;
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = 256;
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = 128;

extern "C" __constant__ uint32_t WARP_SHAPE_M = 24;
extern "C" __constant__ uint32_t WARP_SHAPE_N = 32;
extern "C" __constant__ uint32_t WARP_SHAPE_K = 64;

extern "C" __constant__ uint32_t A_DTYPE_ID = FloatingPointType<16, 8, 7>::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = FloatingPointType<4, 2, 1>::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = FloatingPointType<16, 8, 7>::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = FloatingPointType<8, 4, 3>::kId;

extern "C" __constant__ uint32_t SHAPE_N = 8192;
extern "C" __constant__ uint32_t SHAPE_K = 4096;
extern "C" __constant__ uint32_t PAD_SHAPE_N = 0;
extern "C" __constant__ uint32_t PAD_SHAPE_K = 0;
extern "C" __constant__ uint32_t NUM_EXPERTS = 32;
extern "C" __constant__ uint32_t INPUT_SCALE_GROUP_SIZE = 0;
extern "C" __constant__ uint32_t WEIGHT_SCALE_GROUP_SIZE = 16;
extern "C" __constant__ uint32_t WEIGHT_SCALE_GROUP_SIZE_N = 0;
extern "C" __constant__ uint32_t USE_INT_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t USE_FUSED_E8M0_SCALE = 0;
extern "C" __constant__ uint32_t HAS_ZERO_POINT = 0;
extern "C" __constant__ uint32_t IS_FP_ZERO_POINT = 0;
extern "C" __constant__ uint32_t HAS_BIAS = 0;
extern "C" __constant__ uint32_t USE_PACKED_K_LAYOUT = 0;
extern "C" __constant__ uint32_t MMA_TYPE_ID = 1;
extern "C" __constant__ uint32_t IS_CHANNEL_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t IS_BLOCK_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t IS_GROUP_WEIGHT_SCALE = 1;
extern "C" __constant__ uint32_t IS_TENSOR_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t IS_CHANNEL_WEIGHT_SCALE_2 = 0;
extern "C" __constant__ uint32_t IS_TENSOR_WEIGHT_SCALE_2 = 0;
extern "C" __constant__ uint32_t HAS_CHANNEL_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t HAS_TENSOR_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t HAS_INPUT_SCALE = 0;

extern "C" __constant__ uint32_t USE_F16_ACCUM = 0;
extern "C" __constant__ uint32_t USE_BATCH_INVARIANT = 0;
extern "C" __constant__ uint32_t USE_M_MAJOR_INPUT_SCALE = 0;
extern "C" __constant__ uint32_t GEMM_TYPE_ID = 1;
extern "C" __constant__ uint32_t IS_INDEXED_GEMM = 1;
extern "C" __constant__ uint32_t IS_GROUPED_GEMM = 0;
extern "C" __constant__ uint32_t IS_GROUPED_CONTIGUOUS_GEMM = 0;
extern "C" __constant__ uint32_t IS_GROUPED_MASKED_GEMM = 0;

extern "C" __constant__ uint32_t USE_STREAM_K = 0;
extern "C" __constant__ uint32_t NUM_STAGES = 4;
extern "C" __constant__ uint32_t NUM_CTAS_PER_SM = 1;
extern "C" __constant__ uint32_t USE_WARP_SPEC = 0;
extern "C" __constant__ uint32_t USE_MBARRIER = 0;
extern "C" __constant__ uint32_t USE_CP_ASYNC = 1;
extern "C" __constant__ uint32_t USE_TMA = 0;
extern "C" __constant__ uint32_t USE_TMA_A = 0;
extern "C" __constant__ uint32_t USE_TMA_AS = 0;
extern "C" __constant__ uint32_t USE_TMA_B = 0;
extern "C" __constant__ uint32_t USE_TMA_C = 0;
extern "C" __constant__ uint32_t USE_TMA_BS = 0;
extern "C" __constant__ uint32_t USE_TMA_BS2 = 0;
extern "C" __constant__ uint32_t USE_TMA_BZP = 0;
extern "C" __constant__ uint32_t USE_TMA_BIAS = 0;
extern "C" __constant__ uint32_t REDUCE_OVERLAP_LAST_STAGE_ONLY = 0;
extern "C" __constant__ uint32_t NUM_WRITE_SPLITS = 1;
extern "C" __constant__ uint32_t MULTI_CAST_SIZE_A = 1;
extern "C" __constant__ uint32_t MULTI_CAST_SIZE_B = 1;
extern "C" __constant__ uint32_t RASTER_GROUP_M = 1;
extern "C" __constant__ uint32_t NUM_THREADS = 512;
extern "C" __constant__ uint32_t NUM_MATH_THREADS = 512;
extern "C" __constant__ uint32_t NUM_LOAD_THREADS = 512;
