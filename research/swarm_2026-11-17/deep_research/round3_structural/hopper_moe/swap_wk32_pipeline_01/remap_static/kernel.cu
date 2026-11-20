

#define HUMMING_SHAPE_N 256
#define HUMMING_SHAPE_K 4096
#define HUMMING_PAD_SHAPE_N 0
#define HUMMING_PAD_SHAPE_K 0
#define HUMMING_NUM_EXPERTS 1
#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 16
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE_N 0
#define HUMMING_USE_INT_WEIGHT_SCALE 0
#define HUMMING_USE_FUSED_E8M0_SCALE 0
#define HUMMING_HAS_ZERO_POINT 0
#define HUMMING_IS_FP_ZERO_POINT 0
#define HUMMING_HAS_BIAS 0
#define HUMMING_USE_PACKED_K_LAYOUT 0
#define HUMMING_MMA_TYPE_ID 0
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
#define HUMMING_NUM_STAGES 3
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
#define HUMMING_NUM_THREADS 256
#define HUMMING_NUM_MATH_THREADS 256
#define HUMMING_NUM_LOAD_THREADS 256

#if 0
#include <humming/kernel/humming_ws.cuh>
#else
#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#pragma once

#include <humming/utils/all.cuh>

#include <humming/epilogue/gmem_writer.cuh>
#include <humming/epilogue/smem_reducer.cuh>
#pragma once

#include <humming/utils/all.cuh>


CUDA_INLINE void shlf_trans_mma_c_32b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t val;
  uint32_t idx = (threadIdx.x / 4) % 2;
  switch (idx) {
    case 0: {
      val = vals_uint_ptr[1];
      break;
    };
    case 1: {
      val = vals_uint_ptr[0];
      break;
    }
  }

  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);

  switch (idx) {
    case 0: {
      vals_uint_ptr[1] = swapped_val;
      break;
    };
    case 1: {
      vals_uint_ptr[0] = swapped_val;
      break;
    }
  }
}


CUDA_INLINE void shlf_trans_mma_c_16b(void *vals_ptr) {
  uint32_t *vals_uint_ptr = reinterpret_cast<uint32_t *>(vals_ptr);

  uint32_t &val = vals_uint_ptr[0];
  uint32_t swapped_val = __shfl_xor_sync(0xffffffff, val, 4);
  uint32_t idx = (threadIdx.x / 4) % 2;

  uint16_t *vals_ushort_ptr = reinterpret_cast<uint16_t *>(&val);
  uint16_t *swapped_vals_ushort_ptr = reinterpret_cast<uint16_t *>(&swapped_val);

  switch (idx) {
    case 0: {
      vals_ushort_ptr[1] = swapped_vals_ushort_ptr[0];
      break;
    };
    case 1: {
      vals_ushort_ptr[0] = swapped_vals_ushort_ptr[1];
      break;
    }
  }
}


template <typename T>
CUDA_INLINE void shlf_trans_mma_c(T &vals) {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  if constexpr (sizeof(T) == 8) {
    shlf_trans_mma_c_32b(&vals);
  } else {
    shlf_trans_mma_c_16b(&vals);
  }
}


template <class Ctx, class MMA, class ArithClass>
class EpilogueSmemWriter : F16Conversion<typename Ctx::ElementC> {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementC = typename Ctx::ElementC;

  static constexpr bool kUseWgmma = true; // Local C-fragment orientation only; Ctx stays MMA.

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using MmaShape = typename MmaOpClass::MmaShape;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using CRegistersArrayType = typename MMA::CRegistersArrayType;

  static constexpr uint32_t kNumWriteSplits = Ctx::kNumWriteSplits;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kUseIntWeightScale = Ctx::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kHasGroupScale = kIsGroupInputScale || kIsGroupWeightScale || kIsBlockWeightScale;
  static constexpr bool kIsIntAccum = std::is_same<ValTypeC, int32_t>::value && (!kHasGroupScale || kUseIntWeightScale || kUseFusedE8m0Scale);

  static constexpr uint32_t M_WARPS = BlockShape::M / WarpShape::M;
  static constexpr uint32_t N_WARPS = BlockShape::N / WarpShape::N;
  static constexpr uint32_t K_WARPS = BlockShape::K / WarpShape::K;

public:
  Ctx &ctx;
  ArithClass &arith;

  CUDA_INLINE
  EpilogueSmemWriter(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), arith(arith) {
  }

  CUDA_INLINE
  void write(uint32_t *regs_ptr, uint32_t slice_count, uint32_t split_idx) {
    if (threadIdx.x >= kNumMathThreads / K_WARPS) return;

    auto &regs = *reinterpret_cast<CRegistersArrayType *>(regs_ptr);
    scalar_t2 *smem_half2_ptr = reinterpret_cast<scalar_t2 *>(ctx.smem.reduce);
    uint32_t smem = offsetof(SharedStorage, reduce) / 128 % 8;
    using PackTypeC = std::conditional_t<
        sizeof(ValTypeC) == 2, scalar_t2,
        std::conditional_t<kIsIntAccum, int2, float2>>;

    uint32_t laneid = ctx.lane_id();
    uint32_t warpid = ctx.warp_id();
    uint32_t warp_delta_row = ctx.m_warp_offset();
    uint32_t n_warp_id = ctx.n_warp_id();
    uint32_t group_warp_id = warpid % 4;
    auto write_to_smem = [&](PackTypeC val, uint32_t row_8x8block, uint32_t col_8x8block) {
      scalar_t2 val_half2;

      static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
      if constexpr (kNumWriteSplits == 2) {
        static_assert(M_WARPS == 1);
        uint32_t m_8x8block = kUseWgmma ? col_8x8block : row_8x8block;
        if (split_idx == 0 && m_8x8block >= BlockShape::M / 8 / 2) return;
        if (split_idx == 1 && m_8x8block < BlockShape::M / 8 / 2) return;
      }

      if constexpr (kUseWgmma) shlf_trans_mma_c(val);
      if constexpr (sizeof(ValTypeC) != 4) {
        val_half2 = val;
      } else if constexpr (kIsIntAccum) {
        float2 val_float2 = {__int2float_rn(val.x), __int2float_rn(val.y)};
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val_float2, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val_float2, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val_float2);
      } else {
        if constexpr (kUseWgmma) {
          arith.may_apply_f32_on_smem_write(val, col_8x8block, row_8x8block);
        } else {
          arith.may_apply_f32_on_smem_write(val, row_8x8block, col_8x8block);
        }
        val_half2 = this->float22num2(val);
      };

      uint32_t &val_uint = *reinterpret_cast<uint32_t *>(&val_half2);
      if constexpr (kUseWgmma) {
        arith.may_apply_on_smem_write(val_uint, col_8x8block, row_8x8block);
        col_8x8block = col_8x8block - BlockShape::M / 8 / 2 * split_idx;
      } else {
        arith.may_apply_on_smem_write(val_uint, row_8x8block, col_8x8block);
        row_8x8block = row_8x8block - BlockShape::M / 8 / 2 * split_idx;
      }

      if constexpr (!kUseWgmma) {
        uint32_t sub_row = laneid / 4;
        uint32_t row = warp_delta_row + 8 * row_8x8block + sub_row;
        uint32_t col = col_8x8block * 4 + WarpShape::N / 2 * n_warp_id;

        row = row + (BlockShape::M / kNumWriteSplits) * (col / 32);
        col = ((col % 32 / 4) ^ ((sub_row + smem) % 8)) * 4 + laneid % 4;

        uint32_t idx = row * 32 + col;
        smem_half2_ptr[idx] = val_half2;
      } else {
        uint32_t sub_row = (laneid % 4) * 2 + (laneid % 8) / 4;
        uint32_t row = warp_delta_row + 8 * col_8x8block + sub_row;

        uint32_t count = (64 / WarpShape::N);
        uint32_t col1 = ((n_warp_id % count * (8 / count) + row_8x8block) ^ ((sub_row + smem) % 8)) * 4 + laneid / 8;
        uint32_t col2 = (n_warp_id / count) * (BlockShape::M / kNumWriteSplits * 64 / 2);
        uint32_t idx = row * 32 + col1 + col2;
        smem_half2_ptr[idx] = val_half2;
      }
    };

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
      PRAGMA_UNROLL
      for (uint32_t j = 0; j < sizeof(regs[0]) / sizeof(regs[0][0]); j++) {
        auto part_regs = reinterpret_cast<PackTypeC *>(&regs[i][j]);
        PRAGMA_UNROLL
        for(uint32_t p=0;p<2;p++)write_to_smem(part_regs[p],2*i+p,j);
      }
    }
  }
};



template <class Ctx, class MMA, class ArithClass>
class EpiloguePipeline {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;

  using SmemReducer = EpilogueSmemReducer<Ctx, MMA>;
  using SmemWriter = EpilogueSmemWriter<Ctx, MMA, ArithClass>;
  using GmemWriter = EpilogueGmemWriter<Ctx, ArithClass>;

  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr uint32_t kNumWriteSplits = Ctx::kNumWriteSplits;

public:
  Ctx &ctx;
  SmemReducer smem_reducer;
  SmemWriter smem_writer;
  GmemWriter gmem_writer;
  ArithClass &arith;
  int32_t *locks;

  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  CUDA_INLINE
  EpiloguePipeline(Ctx &ctx, ArithClass &arith)
      : ctx(ctx), locks(ctx.params.locks), arith(arith),
        smem_reducer(ctx), smem_writer(ctx, arith), gmem_writer(ctx, arith) {
    if constexpr (Ctx::kUseTmaC) {
      if constexpr (kIsGroupedGemm) gmem_writer.update_tensor_map_ptr(ctx.params.tensor_map_buffer + blockIdx.x);
      else if (threadIdx.x == 0) prefetch_tensor_map(ctx.params.c);
    }
    ctx.sync_math_threads();
  }

  CUDA_INLINE
  void call(uint32_t *regs_c_ptr) {
    ctx.sync_math_threads();
    if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);
    static_assert(kNumWriteSplits == 1 || kNumWriteSplits == 2);
    if constexpr (kNumWriteSplits > 1) {
      static_assert(BlockShape::M == WarpShape::M);
      static_assert(BlockShape::M % 32 == 0);
      static_assert(!Ctx::kUseTmaC);
    }

    if (slice_count > 1) acquire_gmem_barrier();
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kNumWriteSplits; i++) {
      smem_writer.write(regs_c_ptr, slice_count, i);
      if constexpr (Ctx::kUseTmaC) {
        if (ctx.is_math_thread()) tma_fence_async_shared();
      }
      ctx.sync_math_threads();
      gmem_writer.write(slice_id, slice_count, i);
      ctx.sync_math_threads();
    }
    if (slice_count > 1) release_gmem_barrier();
  }

  CUDA_INLINE
  void acquire_gmem_barrier() {
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 0 : -1;
      barrier_acquire2<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_acquire<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id);
    }
  }

  CUDA_INLINE
  void release_gmem_barrier() {
    if (Ctx::kUseTmaC || slice_count > 3) {
      int32_t val = slice_id == 0 ? 1 - static_cast<int32_t>(slice_count) : 0;
      barrier_release2<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], val);
    } else {
      barrier_release<Ctx::kNumMathThreads, Ctx::kNumThreads>(&locks[locks_offset], slice_id == slice_count - 1);
    }
  }

  CUDA_INLINE
  void seek(uint32_t expert_id, uint32_t m_block_id, uint32_t n_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    gmem_writer.seek(m_block_id, n_block_id, current_shape_m, m_offset);
    if constexpr (Ctx::kHasTensorWeightScale) {
      const uint32_t *gs_ptr = reinterpret_cast<const uint32_t *>(Ctx::kIsTensorWeightScale2 ? ctx.params.bs2 : ctx.params.bs);
      arith.gs = gs_ptr[Ctx::kIsDenseGemm ? 0 : expert_id];
    }
  };

  CUDA_INLINE
  void set_streamk_state(uint32_t slice_count_, uint32_t slice_id_, uint32_t locks_offset_) {
    slice_count = slice_count_;
    slice_id = slice_id_;
    locks_offset = locks_offset_;
  };
};

#include <humming/memory/g2s_pipeline.cuh>
#pragma once

#include <humming/memory/s2r_loader/loader_a.cuh>
#include <humming/memory/s2r_loader/loader_as.cuh>
#include <humming/memory/s2r_loader/loader_b.cuh>
#include <humming/memory/s2r_loader/loader_bias.cuh>
#include <humming/memory/s2r_loader/loader_bs.cuh>
#include <humming/memory/s2r_loader/loader_bzp.cuh>
#include <humming/utils/all.cuh>

template <class Ctx, class MMA, class Epilogue>
class S2RMemoryPipeline {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr uint32_t kPartMmaShapeK = Ctx::kPartMmaShapeK;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsChannelInputScale = kHasInputScale && Ctx::kInputScaleGroupSize == 0;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsChannelWeightScale2 = Ctx::kIsChannelWeightScale2;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kHasBias = Ctx::kHasBias;

  using LoaderA = S2RMemoryLoaderA<Ctx>;
  using LoaderB = S2RMemoryLoaderB<Ctx>;
  using LoaderAS = S2RMemoryLoaderAS<Ctx>;
  using LoaderBS = S2RMemoryLoaderBS<Ctx>;
  using LoaderBZP = S2RMemoryLoaderBZP<Ctx>;
  using LoaderBias = S2RMemoryLoaderBias<Ctx>;

public:
  Ctx &ctx;
  MMA &mma;
  Epilogue &epilogue;
  LoaderA loader_a;
  LoaderB loader_b;
  LoaderAS loader_as;
  LoaderBS loader_bs;
  LoaderBZP loader_bzp;
  LoaderBias loader_bias;

  CUDA_INLINE
  S2RMemoryPipeline(Ctx &ctx, MMA &mma, Epilogue &epilogue)
      : ctx(ctx), mma(mma), epilogue(epilogue),
        loader_a(ctx), loader_b(ctx), loader_as(ctx),
        loader_bs(ctx), loader_bzp(ctx), loader_bias(ctx) {
  }

  template <bool kIsFirst = false>
  CUDA_INLINE void load_stage_iter(uint32_t stage_id, uint32_t iter_id) {
    stage_id = (stage_id + iter_id / Ctx::kWarpIters) % kNumStages;
    iter_id = iter_id % Ctx::kWarpIters;
    uint32_t buffer_id = iter_id % 2;
    uint32_t k_iter_id = Ctx::kUsePackedKLayout ? 0 : iter_id;
    auto &smem = ctx.smem;

    loader_b.load(smem.stages[stage_id].b, mma.regs_qb_as_ptr(buffer_id), iter_id);
    if constexpr (!kUseWgmma)
      mma.load_activation(smem.stages[stage_id].a, buffer_id, iter_id, stage_id);
    if constexpr (kUseMxmma) {
      if constexpr (kIsGroupInputScale)
        loader_as.load_sf(smem.stages[stage_id].as, mma.regs_sfa_as_ptr(buffer_id), k_iter_id);
      if constexpr (kIsGroupOrBlockWeightScale)
        loader_bs.load_sf(smem.stages[stage_id].bs, mma.regs_sfb_as_ptr(buffer_id), k_iter_id);
    } else {
      if constexpr (kIsGroupInputScale)
        loader_as.load(smem.stages[stage_id].as, mma.arith.regs_as_as_ptr(buffer_id), k_iter_id);
      if constexpr (kIsGroupOrBlockWeightScale)
        loader_bs.load(smem.stages[stage_id].bs, mma.arith.regs_bs_as_ptr(buffer_id), k_iter_id);
    }
    if constexpr (kHasZeroPoint && (kIsGroupOrBlockWeightScale || kIsFirst)) {
      if constexpr (kIsChannelWeightScale)
        loader_bzp.load(smem.bzp_c, mma.arith.regs_zp_as_ptr(buffer_id), k_iter_id);
      else
        loader_bzp.load(smem.stages[stage_id].bzp, mma.arith.regs_zp_as_ptr(buffer_id), k_iter_id);
    }
  }

  CUDA_INLINE void load_channel(uint32_t slice_id) {
    auto &smem = ctx.smem;
    if constexpr (kIsChannelInputScale) loader_as.load(smem.as_c, epilogue.arith.regs_as_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale) loader_bs.load(smem.bs_c, epilogue.arith.regs_bs_as_ptr(), -1);
    if constexpr (kIsChannelWeightScale2) loader_bias.load(smem.bs2_c, epilogue.arith.regs_bs2_as_ptr(), 1);
    if constexpr (kHasBias) loader_bias.load(smem.bias, epilogue.arith.regs_bias_as_ptr(), slice_id == 0);
  }
};

#include <humming/mma/all.cuh>
// Isolated swap-AB MMA adapter. Original cp.async/scheduler/arithmetic/K reducer remain pinned.
template<class Ctx,class ArithClass> struct ResearchSwapMMA {
 using MmaOpClass=typename Ctx::MmaOpClass;
 using WarpShape=typename Ctx::WarpShape;
 using BlockShape=typename Ctx::BlockShape;
 using SharedStorage=typename Ctx::SharedStorage;
 using ElementA=typename Ctx::ElementA;
 using ElementB=typename Ctx::ElementB;
 using CRegistersType=typename MmaOpClass::CRegisters;
 using CRegistersArrayType=CRegistersType[WarpShape::N/16][WarpShape::M/8];
 static_assert(!Ctx::kUseWgmma && !Ctx::kUseWarpSpec && !Ctx::kUseTmaA && !Ctx::kUseTmaB && !Ctx::kUseTmaC);
 static_assert(Ctx::kIsIndexedGemm && Ctx::kUseCpAsync && !Ctx::kUseStreamK);
 static_assert(WarpShape::N==64 && BlockShape::M==WarpShape::M);
 static_assert((BlockShape::K==64 && (WarpShape::K==32 || WarpShape::K==64)) || (BlockShape::K==128 && WarpShape::K==64));
 static_assert(WarpShape::M==8 || WarpShape::M==16 || WarpShape::M==24);
 static_assert(ElementA::kBits==16 && ElementB::kBits==4 && Ctx::kWeightScaleGroupSize==16);
 static_assert(!Ctx::kHasZeroPoint && !Ctx::kHasBias && Ctx::kInputScaleGroupSize==0);
 Ctx &ctx;ArithClass &arith;
 uint32_t regs_qb[2][4];
 uint32_t regs_weight_a[2][WarpShape::N/16][4];
 uint32_t regs_act_b[2][WarpShape::M/8][2];
 CRegistersArrayType regs_c;
 CUDA_INLINE ResearchSwapMMA(Ctx &c,ArithClass&a):ctx(c),arith(a){}
 CUDA_INLINE void zero_accum(){
  uint32_t*p=reinterpret_cast<uint32_t*>(regs_c);
  PRAGMA_UNROLL
  for(uint32_t i=0;i<sizeof(regs_c)/4;i++)p[i]=0;
 }
 CUDA_INLINE void transform_b(uint32_t buffer,uint32_t iter){
  uint32_t standard_qb[4];
  PRAGMA_UNROLL
  for(uint32_t i=0;i<4;i++){
   uint32_t w=regs_qb[buffer][i];
   standard_qb[i]=(w&0xf00ff00fu)|((w&0x00f000f0u)<<4)|((w&0x0f000f00u)>>4);
  }
  PRAGMA_UNROLL
  for(uint32_t i=0;i<WarpShape::N/16;i++){
   uint32_t b[4];uint32_t zp[4]={0,0,0,0};
   if constexpr(false){
    dequant<ElementB,ElementA,false,false,1>(regs_qb[buffer],b,i,zp);
    arith.may_apply_bs_and_zp_on_b(b,i,buffer);
    PRAGMA_UNROLL
    for(uint32_t r=0;r<4;r++)regs_weight_a[buffer][i][r]=b[r];
   }else{
    dequant<ElementB,ElementA,false,false,1>(standard_qb,b,i,zp);
    arith.may_apply_bs_and_zp_on_b(b,i,buffer);
    regs_weight_a[buffer][i][0]=b[0];regs_weight_a[buffer][i][1]=b[2];
    regs_weight_a[buffer][i][2]=b[1];regs_weight_a[buffer][i][3]=b[3];
   }
  }
 }
 CUDA_INLINE void load_activation(const int4*smem_ptr,uint32_t buffer,uint32_t iter,uint32_t stage){
  uint32_t base=(offsetof(SharedStorage,stages)+stage*sizeof(typename SharedStorage::StageStorage))/128%8;
  PRAGMA_UNROLL
  for(uint32_t t=0;t<WarpShape::M/8;t++){
   PRAGMA_UNROLL
   for(uint32_t reg=0;reg<2;reg++){
    uint32_t row=ctx.m_warp_offset()+t*8+ctx.lane_id()/4;
    uint32_t col=iter*2+ctx.k_warp_offset()/8+reg;
    row=BlockShape::M*(col/8)+row;
    col=(col%8)^((row+base)%8);
    const uint32_t*source=reinterpret_cast<const uint32_t*>(smem_ptr+row*8+col);
    regs_act_b[buffer][t][reg]=source[ctx.lane_id()%4];
   }
  }
 }
 CUDA_INLINE void run(uint32_t stage,uint32_t iter){
  uint32_t buffer=iter%2;
  PRAGMA_UNROLL
  for(uint32_t t=0;t<WarpShape::M/8;t++){
   if constexpr(false){
    if(t>0 && ctx.smem.wr_row_index[ctx.m_warp_offset()+t*8]>=ctx.params.shape_m*ctx.params.top_k)continue;
   }
   PRAGMA_UNROLL
   for(uint32_t n=0;n<WarpShape::N/16;n++)MmaOpClass::fma(regs_weight_a[buffer][n],regs_act_b[buffer][t],regs_c[n][t],regs_c[n][t]);
  }
 }
 template<class T=uint32_t>CUDA_INLINE T*regs_qb_as_ptr(uint32_t b){return reinterpret_cast<T*>(regs_qb[b]);}
 template<class T=uint32_t>CUDA_INLINE T*regs_c_as_ptr(uint32_t b=0){return reinterpret_cast<T*>(regs_c);}
 template<class T=uint32_t>CUDA_INLINE T*final_regs_c_as_ptr(){return reinterpret_cast<T*>(regs_c);}
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
  using MMA = ResearchSwapMMA<Ctx, MainloopArithmetic>;
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
  static constexpr MmaType kMmaType = MmaType::MMA;
  using MmaShape = Shape<16, 8, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using DRegisters = float[4];

  CUDA_INLINE
  static void fma(uint32_t *a, uint32_t *b, float *c, float *d) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
         "r"(b[0]),  "r"(b[1]),
         "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
  };
};

class LayerConfig {
public:
  static constexpr auto kShapeN = 256u;
  static constexpr auto kShapeK = 4096u;
  static constexpr auto kPadShapeN = 0u;
  static constexpr auto kPadShapeK = 0u;
  static constexpr auto kNumExperts = 1u;
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
  static constexpr auto kMmaType = MmaType::MMA;
  static constexpr auto kUsePackedKLayout = false;
  static constexpr auto kMmaTypeId = 0u;
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
  static constexpr auto kNumStages = 3u;
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
  static constexpr auto kNumThreads = 256u;
  static constexpr auto kNumMathThreads = 256u;
  static constexpr auto kNumLoadThreads = 256u;
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<16, 256, 64>,
    Shape<16, 64, 32>,
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

extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = 256;
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = 4096;

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = 16;
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = 256;
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = 64;

extern "C" __constant__ uint32_t WARP_SHAPE_M = 16;
extern "C" __constant__ uint32_t WARP_SHAPE_N = 64;
extern "C" __constant__ uint32_t WARP_SHAPE_K = 32;

extern "C" __constant__ uint32_t A_DTYPE_ID = FloatingPointType<16, 8, 7>::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = FloatingPointType<4, 2, 1>::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = FloatingPointType<16, 8, 7>::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = FloatingPointType<8, 4, 3>::kId;

extern "C" __constant__ uint32_t SHAPE_N = 256;
extern "C" __constant__ uint32_t SHAPE_K = 4096;
extern "C" __constant__ uint32_t PAD_SHAPE_N = 0;
extern "C" __constant__ uint32_t PAD_SHAPE_K = 0;
extern "C" __constant__ uint32_t NUM_EXPERTS = 1;
extern "C" __constant__ uint32_t INPUT_SCALE_GROUP_SIZE = 0;
extern "C" __constant__ uint32_t WEIGHT_SCALE_GROUP_SIZE = 16;
extern "C" __constant__ uint32_t WEIGHT_SCALE_GROUP_SIZE_N = 0;
extern "C" __constant__ uint32_t USE_INT_WEIGHT_SCALE = 0;
extern "C" __constant__ uint32_t USE_FUSED_E8M0_SCALE = 0;
extern "C" __constant__ uint32_t HAS_ZERO_POINT = 0;
extern "C" __constant__ uint32_t IS_FP_ZERO_POINT = 0;
extern "C" __constant__ uint32_t HAS_BIAS = 0;
extern "C" __constant__ uint32_t USE_PACKED_K_LAYOUT = 0;
extern "C" __constant__ uint32_t MMA_TYPE_ID = 0;
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
extern "C" __constant__ uint32_t NUM_STAGES = 3;
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
extern "C" __constant__ uint32_t NUM_THREADS = 256;
extern "C" __constant__ uint32_t NUM_MATH_THREADS = 256;
extern "C" __constant__ uint32_t NUM_LOAD_THREADS = 256;
