

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
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>
#pragma once

#include <humming/utils/all.cuh>


template <uint32_t swizzle_bytes = 128>
CUDA_INLINE uint64_t research_wait1_make_smem_desc(uint32_t addr) {
  static_assert(swizzle_bytes == 128 || swizzle_bytes == 64);

  constexpr uint64_t swizzle_type = swizzle_bytes == 128 ? 1 : 2;
  constexpr uint64_t stride = (swizzle_bytes * 8) >> 4;
  constexpr uint64_t desc_base = (swizzle_type << 62) | (stride << 32);

  uint64_t desc = desc_base;
  reinterpret_cast<uint32_t *>(&desc)[0] = (addr >> 4);

  return desc;
};


template <class Ctx, class ArithClass>
struct ResearchWait1WGMMA {
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
  ResearchWait1WGMMA(Ctx &ctx, ArithClass &arith)
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
        dequant<ElementB, ElementA, kHasZeroPoint, kIsFpZeroPoint, kNumWarpShapeNSplits>(regs_qb[buffer_id], regs_b_ptr, i, zp_vals_ptr);
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
      uint64_t desc = research_wait1_make_smem_desc<kSwizzleBytes>(smem_addr);

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
    wgmma_wait<1>();
    may_fence_regs(delta_j);

    PRAGMA_UNROLL
    for (uint32_t k = 0; k < kRunKLoop; k++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < kNumIters; j++) {
        arith.may_apply_as_and_bs_on_wgmma_c(regs_c_as_ptr(), j, k, iter_id, delta_m);
      }
    }
  };

  CUDA_INLINE void drain_async() {
    wgmma_wait<0>();
    // Prevent compiler movement of accumulator consumption above the final wait.
    fence_regs(regs_c[0]);
  }

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
  using MMA = ResearchWait1WGMMA<Ctx, MainloopArithmetic>;
  static_assert(Ctx::kUseWgmma && Ctx::kIsIndexedGemm && !Ctx::kUsePackedKLayout);
  static_assert(ElementA::kBits == 16 && ElementB::kBits == 4 && ElementBS::kBits == 8);
  static_assert(Ctx::kWeightScaleGroupSize == 16 && Ctx::kIsGroupWeightScale);
  static_assert(!Ctx::kUseFusedE8m0Scale && !Ctx::kHasZeroPoint && !Ctx::kHasBias);
  static_assert(!TuningConfig::kUseWarpSpec && !TuningConfig::kUseStreamK && !ComputeConfig::kUseF16Accum);
  static_assert(kNumStages == 3 || kNumStages == 4);
  static_assert(WarpShape::K == 64 && Ctx::kWarpIters == 4);
  static_assert(TuningConfig::kMultiCastSizeA == 1 && TuningConfig::kMultiCastSizeB == 1);
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

    mma.drain_async();
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
