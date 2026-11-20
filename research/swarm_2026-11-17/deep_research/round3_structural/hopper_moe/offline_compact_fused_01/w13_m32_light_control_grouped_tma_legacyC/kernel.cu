

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
#define HUMMING_GEMM_TYPE_ID 2
#define HUMMING_IS_INDEXED_GEMM 0
#define HUMMING_IS_GROUPED_GEMM 1
#define HUMMING_IS_GROUPED_CONTIGUOUS_GEMM 1
#define HUMMING_IS_GROUPED_MASKED_GEMM 0

#define HUMMING_USE_STREAM_K 0
#define HUMMING_NUM_STAGES 4
#define HUMMING_NUM_CTAS_PER_SM 1
#define HUMMING_USE_WARP_SPEC 1
#define HUMMING_USE_MBARRIER 1
#define HUMMING_USE_CP_ASYNC 1
#define HUMMING_USE_TMA 1
#define HUMMING_USE_TMA_A 1
#define HUMMING_USE_TMA_AS 0
#define HUMMING_USE_TMA_B 1
#define HUMMING_USE_TMA_C 0
#define HUMMING_USE_TMA_BS 1
#define HUMMING_USE_TMA_BS2 0
#define HUMMING_USE_TMA_BZP 0
#define HUMMING_USE_TMA_BIAS 0
#define HUMMING_REDUCE_OVERLAP_LAST_STAGE_ONLY 0
#define HUMMING_NUM_WRITE_SPLITS 1
#define HUMMING_MULTI_CAST_SIZE_A 1
#define HUMMING_MULTI_CAST_SIZE_B 1
#define HUMMING_RASTER_GROUP_M 1
#define HUMMING_NUM_THREADS 640
#define HUMMING_NUM_MATH_THREADS 512
#define HUMMING_NUM_LOAD_THREADS 128

#if 1
#pragma once

#pragma once

#include <humming/utils/all.cuh>

template <class Ctx>
class Scheduler {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;

  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kIsDenseGemm = Ctx::kIsDenseGemm;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr bool kIsGroupedContiguousGemm = Ctx::kIsGroupedContiguousGemm;
  static constexpr uint32_t kNumExperts = Ctx::kNumExperts;
  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  static constexpr uint32_t kInputScaleGroupSize = Ctx::kInputScaleGroupSize > 0 ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr uint32_t kAsBlocksPerWord = kUseMxmma ? MAX(1u, 4 * kInputScaleGroupSize / BlockShape::K) : 1;

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;

  static constexpr uint32_t kRasterGroupM = Ctx::kRasterGroupM;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr int32_t ct_gcd(int32_t a, int32_t b) { return b == 0 ? a : ct_gcd(b, a % b); }

  uint32_t m_blocks;
  uint32_t mn_blocks;
  uint32_t mnk_blocks;

  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;
  uint32_t dp_mn_total_iters;
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;

public:
  Ctx &ctx;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t old_m_block_id = 0;

  // for stream-k
  uint32_t slice_iters;
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  // for tma multi-cast
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;

  // for moe gemm (indexed gemm or grouped gemm)
  uint32_t old_expert_id = (1 << 30);
  uint32_t expert_id = 0;

  // for grouped gemm
  uint32_t current_shape_m;
  uint32_t expert_max_num_tokens;
  uint32_t offset_in_expert = 0;
  uint32_t m_offset = 0;

  CUDA_INLINE
  Scheduler(Ctx &ctx) : ctx(ctx) {

    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x == 0) ctx.smem.tensor_map_buffer[0] = reinterpret_cast<const CUtensorMap *>(ctx.params.c)[0];
      __syncwarp();
    }

    current_shape_m = ctx.params.shape_m;
    expert_max_num_tokens = ctx.params.shape_m / Ctx::kNumExperts;
    calc_m_blocks();
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * K_BLOCKS;
    uint32_t kNumCtaGroups = gridDim.x / kMultiCastSize;

    if constexpr (kUseStreamK) {
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > kNumCtaGroups) {
        streamk_mn_blocks = mn_blocks % kNumCtaGroups;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= kNumCtaGroups) streamk_mn_blocks += kNumCtaGroups;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / kNumCtaGroups;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * K_BLOCKS;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, kNumCtaGroups);

      constexpr int32_t blocks_per_group = MAX(kMaxGroupSize / BlockShape::K, kAsBlocksPerWord);
      constexpr int32_t bpg = blocks_per_group > 1 ? blocks_per_group : 1;
      constexpr int32_t align_iters = bpg / ct_gcd(bpg, (int32_t)kNumStages) * (int32_t)kNumStages;
      if constexpr (align_iters > 1) {
        streamk_mnk_total_iters = align_iters * CEIL_DIV(streamk_mnk_total_iters, align_iters);
      };

      streamk_mnk_next_index = kNumCtaGroups * dp_mn_iters * K_BLOCKS + streamk_mnk_total_iters * (blockIdx.x / kMultiCastSize);

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_iters = mn_blocks / kNumCtaGroups;
      if (blockIdx.x / kMultiCastSize < mn_blocks % kNumCtaGroups) { dp_mn_iters += 1; };
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;

    if (dp_mn_iters) { dp_mn_next_index = blockIdx.x / kMultiCastSize; };
  };

  CUDA_INLINE
  void calc_m_blocks() {
    if constexpr (kIsDenseGemm) {
      m_blocks = CEIL_DIV(ctx.params.shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsIndexedGemm) {
      uint32_t padded_shape_m = ctx.params.num_tokens_padded_ptr[0];
      m_blocks = CEIL_DIV(padded_shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsGroupedGemm) {
      auto &expert_layout = ctx.params.expert_layout_ptr;
      if constexpr (kIsGroupedContiguousGemm) {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_offset);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_offset);
      } else {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_tokens);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_tokens);
      }
      if constexpr (kUseCpAsync) cp_async_commit_group();
      if constexpr (kUseCpAsync) cp_async_wait_group<0>();
      __syncthreads();

      if constexpr (kIsGroupedContiguousGemm) {
        ctx.smem.expert_tokens[kNumExperts - 1] = ctx.smem.expert_offset[kNumExperts] - ctx.smem.expert_offset[kNumExperts - 1];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts - 1, kNumThreads); i++) {
          uint32_t index = kNumThreads * i + threadIdx.x;
          if (index < kNumExperts - 1) {
            ctx.smem.expert_tokens[index] = ctx.smem.expert_offset[index + 1] - ctx.smem.expert_offset[index];
          }
        }

        __syncthreads();
      }

      if (ctx.warp_id() == 0) {
        uint32_t tmp_m_blocks = 0;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts, 32); i++) {
          uint32_t index = 32 * i + threadIdx.x;
          if (index < kNumExperts) {
            tmp_m_blocks += CEIL_DIV(ctx.smem.expert_tokens[index], BlockShape::M);
          }
        }

        __syncwarp();
        m_blocks = warp_reduce_add(tmp_m_blocks);
        if (threadIdx.x == 0) ctx.smem.total_m_blocks[0] = m_blocks;
      }

      __syncthreads();
      m_blocks = ctx.smem.total_m_blocks[0];
    }
  }

  CUDA_INLINE
  void map_mn_block(uint32_t mn_index, uint32_t &m_id, uint32_t &n_id) {
    if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {
      const uint32_t blocks_per_group = kRasterGroupM * N_BLOCKS;
      const uint32_t group_id = mn_index / blocks_per_group;
      const uint32_t first_m = group_id * kRasterGroupM;
      const uint32_t group_m = MIN(m_blocks - first_m, kRasterGroupM);
      const uint32_t idx_in_group = mn_index - group_id * blocks_per_group;
      m_id = first_m + idx_in_group % group_m;
      n_id = idx_in_group / group_m;
    }
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      slice_iters = K_BLOCKS;

      map_mn_block(dp_mn_next_index, m_block_id, n_block_id);

      if constexpr (kMultiCastSizeB > 1) {
        m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
      } else if constexpr (kMultiCastSizeA > 1) {
        n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
      }
      k_block_id = 0;
      dp_mn_next_index += gridDim.x / kMultiCastSize;
      dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if constexpr (kIsIndexedGemm) {
      if (has_next_block) fetch_moe_index_block();
    }
    if constexpr (kIsGroupedGemm) {
      if (has_next_block) fetch_moe_group_block();
    }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / K_BLOCKS;

    map_mn_block(streamk_mn_index, m_block_id, n_block_id);
    if constexpr (kMultiCastSizeB > 1) {
      m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
    } else if constexpr (kMultiCastSizeA > 1) {
      n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
    }
    k_block_id = streamk_mnk_next_index - streamk_mn_index * K_BLOCKS;

    slice_iters = K_BLOCKS - k_block_id;
    slice_iters = slice_iters > streamk_mnk_iters ? streamk_mnk_iters : slice_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x / kMultiCastSize;
    locks_offset = locks_offset * kMultiCastSize + cluster_rank;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_group_block() {
    uint32_t delta_m_block_id = m_block_id - old_m_block_id;
    offset_in_expert += delta_m_block_id * BlockShape::M;

    while (offset_in_expert >= ctx.smem.expert_tokens[expert_id]) {
      offset_in_expert -= CEIL_DIV(ctx.smem.expert_tokens[expert_id], BlockShape::M) * BlockShape::M;
      expert_id++;
    }

    old_m_block_id = m_block_id;
    if constexpr (Ctx::kIsGroupedMaskedGemm) {
      m_offset = expert_id * expert_max_num_tokens + offset_in_expert;
      current_shape_m = expert_id * expert_max_num_tokens + ctx.smem.expert_tokens[expert_id];
    } else if constexpr (Ctx::kIsGroupedContiguousGemm) {
      m_offset = ctx.smem.expert_offset[expert_id] + offset_in_expert;
      current_shape_m = ctx.smem.expert_offset[expert_id] + ctx.smem.expert_tokens[expert_id];
    }

    if (old_expert_id != expert_id) update_tensor_map_c();
    old_expert_id = expert_id;
  };

  CUDA_INLINE
  void fetch_moe_index_block() {
    if (kUseWarpSpec && ctx.is_math_thread()) return;

    expert_id = ctx.params.expert_ids_ptr[m_block_id];

    const uint32_t *gmem_ptr = ctx.params.sorted_ids_ptr + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(ctx.smem.wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumLoadThreads, kLoadThreadOffset>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    ctx.sync_load_threads();

    uint32_t thread_id = threadIdx.x;
    if constexpr (kUseWarpSpec) thread_id = thread_id - kNumMathThreads;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t index = kNumLoadThreads * i + thread_id;
      if (index < BlockShape::M) {
        uint32_t idx = ctx.smem.wr_row_index[index];
        ctx.smem.rd_row_index[index] = idx / ctx.params.top_k;
      };
    }

    ctx.sync_load_threads();
  };

  CUDA_INLINE
  void update_tensor_map_c() {
    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x < 32) {
        __syncwarp();
        if (threadIdx.x == 0) {
          tensor_map_replace_global_dim<1>(ctx.smem.tensor_map_buffer, current_shape_m);
          ctx.params.tensor_map_buffer[blockIdx.x] = ctx.smem.tensor_map_buffer[0];
          tensor_map_release_cta();
          tensor_map_acquire_cta(ctx.params.tensor_map_buffer + blockIdx.x);
        }
        __syncwarp();
      }
    }
  }
};

#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

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
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  constexpr bool kReduceOverlapLastStageOnly = TuningConfig::kReduceOverlapLastStageOnly;
  constexpr uint32_t kLoadThreadRegisters = TuningConfig::kNumMathThreads > 256 || (TuningConfig::kNumCtasPerSm == 1 && ElementA::kBits != 16) ? 40 : 24;

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
  constexpr bool kUseTwoStageReduceBarrier = SharedStorage::kUseTwoStageReduceBarrier;
  static_assert(Ctx::kWarpIters >= 2, "warp-specialized mainloop requires at least two warp iterations");

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
  if (ctx.is_load_thread()) ProducerPipeline::init_mbarrier(ctx);

  mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();

  if (ctx.is_load_thread()) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kLoadThreadRegisters));

    auto producer = ProducerPipeline(ctx);
    if constexpr (Ctx::kIsIndexedGemm) producer.wait_math_epilogue();
    while (scheduler.get_next_block()) {
      debug_kernel_timeout_check(debug_start_clock);
      uint32_t &slice_iters = scheduler.slice_iters;

      producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
      if constexpr (!Ctx::kIsIndexedGemm) producer.wait_math_epilogue();
      producer.load_stage<true, true>(0);
      if constexpr (kUseTwoStageReduceBarrier) producer.wait_reduce_epilogue();
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
        producer.load_stage(stage_id, stage_id < slice_iters);
      };

      while (slice_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          debug_kernel_timeout_check(debug_start_clock);
          if (slice_iters == 1) producer.load_channel();
          producer.wait_stage(stage_id);
          if constexpr (kNumStages == 2) {
            producer.load_stage(stage_id, slice_iters > kNumStages);
          } else {
            producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
          }
          slice_iters--;
          if (!slice_iters) break;
        }
      }
      if constexpr (Ctx::kIsIndexedGemm) producer.wait_math_epilogue();
    }
  } else {
    constexpr uint32_t kPreferredMathThreadRegisters = TuningConfig::kNumMathThreads > 256 ? 96 : 232;
    constexpr uint32_t kNumWarps = TuningConfig::kNumThreads / 32;
    constexpr uint32_t kRegisterAllocationGranularityPerWarp = 256;
    constexpr uint32_t kRegisterBudgetPerWarp =
        (64 * 1024) / (kNumWarps * TuningConfig::kNumCtasPerSm) /
        kRegisterAllocationGranularityPerWarp * kRegisterAllocationGranularityPerWarp;
    constexpr uint32_t kRegisterBudgetPerCta = kRegisterBudgetPerWarp * kNumWarps;
    constexpr uint32_t kLoadThreadRegisterUsage = TuningConfig::kNumLoadThreads * kLoadThreadRegisters;
    constexpr uint32_t kRegistersAvailableForMath = kRegisterBudgetPerCta > kLoadThreadRegisterUsage ? kRegisterBudgetPerCta - kLoadThreadRegisterUsage : 0;
    constexpr uint32_t kMathThreadRegisters = MIN(kPreferredMathThreadRegisters, MAX(24, kRegistersAvailableForMath / TuningConfig::kNumMathThreads / 8 * 8));
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kMathThreadRegisters));

    auto mainloop_arith = MainloopArithmetic();
    auto epilogue_arith = EpilogueArithmetic();
    auto mma = MMA(ctx, mainloop_arith);
    auto epilogue = Epilogue(ctx, epilogue_arith);
    auto consumer = ConsumerPipeline(ctx);
    auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

    consumer.arrive(kNumStages);
    if constexpr (kUseTwoStageReduceBarrier) consumer.arrive(kNumStages + 1);

    while (scheduler.get_next_block()) {
      debug_kernel_timeout_check(debug_start_clock);
      mma.zero_accum();

      uint32_t &slice_iters = scheduler.slice_iters;
      epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
      epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

      consumer.wait_stage<true>(kNumStages);
      s2r_pipe.load_stage_iter<true>(0, 0);
      mma.transform_b(0, 0);

      while (slice_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          debug_kernel_timeout_check(debug_start_clock);
          PRAGMA_UNROLL
          for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
            s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
            mma.run(stage_id, warp_iter_id);
            if (warp_iter_id == Ctx::kWarpIters - 2) {
              consumer.arrive(stage_id);
              if (slice_iters > 1) {
                consumer.wait_stage((stage_id + 1) % kNumStages);
              }
            }
            mma.transform_b(
                (warp_iter_id + 1) % 2,
                (warp_iter_id + 1) % Ctx::kWarpIters);
          }

          slice_iters--;
          if (!slice_iters) break;
        };
      };

      consumer.wait_channel();
      s2r_pipe.load_channel(scheduler.slice_id);

      if constexpr (kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
      epilogue.call(mma.final_regs_c_as_ptr());
      if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
      if constexpr (kUseTwoStageReduceBarrier) consumer.arrive(kNumStages + 1);
      if constexpr (!kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
    }
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB > 1) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};

#else
#pragma once

#pragma once

#include <humming/utils/all.cuh>

template <class Ctx>
class Scheduler {
private:
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;

  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseStreamK = Ctx::kUseStreamK;
  static constexpr bool kIsDenseGemm = Ctx::kIsDenseGemm;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;
  static constexpr bool kIsGroupedContiguousGemm = Ctx::kIsGroupedContiguousGemm;
  static constexpr uint32_t kNumExperts = Ctx::kNumExperts;
  static constexpr uint32_t kNumThreads = Ctx::kNumThreads;
  static constexpr uint32_t kNumMathThreads = Ctx::kNumMathThreads;
  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = kNumThreads - kNumLoadThreads;
  static constexpr uint32_t kMultiCastSizeA = Ctx::kMultiCastSizeA;
  static constexpr uint32_t kMultiCastSizeB = Ctx::kMultiCastSizeB;
  static constexpr uint32_t kMultiCastSize = kMultiCastSizeA * kMultiCastSizeB;

  static constexpr uint32_t kInputScaleGroupSize = Ctx::kInputScaleGroupSize > 0 ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = Ctx::kWeightScaleGroupSize > 0 ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kMaxGroupSize = MAX(kInputScaleGroupSize, kWeightScaleGroupSize);

  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr uint32_t kAsBlocksPerWord = kUseMxmma ? MAX(1u, 4 * kInputScaleGroupSize / BlockShape::K) : 1;

  static constexpr uint32_t N_BLOCKS = ProblemShape::N / BlockShape::N / kMultiCastSizeA;
  static constexpr uint32_t K_BLOCKS = ProblemShape::K / BlockShape::K;

  static constexpr uint32_t kRasterGroupM = Ctx::kRasterGroupM;
  static constexpr uint32_t kNumStages = Ctx::TuningConfig::kNumStages;

  static constexpr int32_t ct_gcd(int32_t a, int32_t b) { return b == 0 ? a : ct_gcd(b, a % b); }

  uint32_t m_blocks;
  uint32_t mn_blocks;
  uint32_t mnk_blocks;

  uint32_t streamk_mnk_total_iters;
  uint32_t streamk_mnk_iters;
  uint32_t streamk_mnk_next_index;
  uint32_t dp_mn_total_iters;
  uint32_t dp_mn_iters;
  uint32_t dp_mn_next_index;

public:
  Ctx &ctx;

  uint32_t m_block_id;
  uint32_t n_block_id;
  uint32_t k_block_id;
  uint32_t old_m_block_id = 0;

  // for stream-k
  uint32_t slice_iters;
  uint32_t slice_count;
  uint32_t slice_id;
  uint32_t locks_offset;

  // for tma multi-cast
  uint32_t cluster_rank = blockIdx.x % kMultiCastSize;

  // for moe gemm (indexed gemm or grouped gemm)
  uint32_t old_expert_id = (1 << 30);
  uint32_t expert_id = 0;

  // for grouped gemm
  uint32_t current_shape_m;
  uint32_t expert_max_num_tokens;
  uint32_t offset_in_expert = 0;
  uint32_t m_offset = 0;

  CUDA_INLINE
  Scheduler(Ctx &ctx) : ctx(ctx) {

    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x == 0) ctx.smem.tensor_map_buffer[0] = reinterpret_cast<const CUtensorMap *>(ctx.params.c)[0];
      __syncwarp();
    }

    current_shape_m = ctx.params.shape_m;
    expert_max_num_tokens = ctx.params.shape_m / Ctx::kNumExperts;
    calc_m_blocks();
    mn_blocks = m_blocks * N_BLOCKS;
    mnk_blocks = mn_blocks * K_BLOCKS;
    uint32_t kNumCtaGroups = gridDim.x / kMultiCastSize;

    if constexpr (kUseStreamK) {
      uint32_t streamk_mn_blocks = mn_blocks;
      if (mn_blocks > kNumCtaGroups) {
        streamk_mn_blocks = mn_blocks % kNumCtaGroups;
        if (streamk_mn_blocks && streamk_mn_blocks * 10 <= kNumCtaGroups) streamk_mn_blocks += kNumCtaGroups;
      }

      dp_mn_iters = (mn_blocks - streamk_mn_blocks) / kNumCtaGroups;

      uint32_t streamk_mnk_blocks = streamk_mn_blocks * K_BLOCKS;

      streamk_mnk_total_iters = CEIL_DIV(streamk_mnk_blocks, kNumCtaGroups);

      constexpr int32_t blocks_per_group = MAX(kMaxGroupSize / BlockShape::K, kAsBlocksPerWord);
      constexpr int32_t bpg = blocks_per_group > 1 ? blocks_per_group : 1;
      constexpr int32_t align_iters = bpg / ct_gcd(bpg, (int32_t)kNumStages) * (int32_t)kNumStages;
      if constexpr (align_iters > 1) {
        streamk_mnk_total_iters = align_iters * CEIL_DIV(streamk_mnk_total_iters, align_iters);
      };

      streamk_mnk_next_index = kNumCtaGroups * dp_mn_iters * K_BLOCKS + streamk_mnk_total_iters * (blockIdx.x / kMultiCastSize);

      if (streamk_mnk_next_index >= mnk_blocks) {
        streamk_mnk_iters = 0;
      } else {
        streamk_mnk_iters = mnk_blocks - streamk_mnk_next_index;
        if (streamk_mnk_iters > streamk_mnk_total_iters) streamk_mnk_iters = streamk_mnk_total_iters;
      };
    } else {
      dp_mn_iters = mn_blocks / kNumCtaGroups;
      if (blockIdx.x / kMultiCastSize < mn_blocks % kNumCtaGroups) { dp_mn_iters += 1; };
    }

    dp_mn_total_iters = dp_mn_iters;
    slice_count = 1;
    slice_id = 0;

    if (dp_mn_iters) { dp_mn_next_index = blockIdx.x / kMultiCastSize; };
  };

  CUDA_INLINE
  void calc_m_blocks() {
    if constexpr (kIsDenseGemm) {
      m_blocks = CEIL_DIV(ctx.params.shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsIndexedGemm) {
      uint32_t padded_shape_m = ctx.params.num_tokens_padded_ptr[0];
      m_blocks = CEIL_DIV(padded_shape_m, BlockShape::M * kMultiCastSizeB);
    } else if constexpr (kIsGroupedGemm) {
      auto &expert_layout = ctx.params.expert_layout_ptr;
      if constexpr (kIsGroupedContiguousGemm) {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_offset);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts + 1, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_offset);
      } else {
        if (ctx.params.use_int64_expert_layout)
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 2, 1>(expert_layout, ctx.smem.expert_tokens);
        else
          legacy_load_2d<kUseCpAsync, kNumExperts, kNumThreads, 1, 1>(expert_layout, ctx.smem.expert_tokens);
      }
      if constexpr (kUseCpAsync) cp_async_commit_group();
      if constexpr (kUseCpAsync) cp_async_wait_group<0>();
      __syncthreads();

      if constexpr (kIsGroupedContiguousGemm) {
        ctx.smem.expert_tokens[kNumExperts - 1] = ctx.smem.expert_offset[kNumExperts] - ctx.smem.expert_offset[kNumExperts - 1];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts - 1, kNumThreads); i++) {
          uint32_t index = kNumThreads * i + threadIdx.x;
          if (index < kNumExperts - 1) {
            ctx.smem.expert_tokens[index] = ctx.smem.expert_offset[index + 1] - ctx.smem.expert_offset[index];
          }
        }

        __syncthreads();
      }

      if (ctx.warp_id() == 0) {
        uint32_t tmp_m_blocks = 0;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumExperts, 32); i++) {
          uint32_t index = 32 * i + threadIdx.x;
          if (index < kNumExperts) {
            tmp_m_blocks += CEIL_DIV(ctx.smem.expert_tokens[index], BlockShape::M);
          }
        }

        __syncwarp();
        m_blocks = warp_reduce_add(tmp_m_blocks);
        if (threadIdx.x == 0) ctx.smem.total_m_blocks[0] = m_blocks;
      }

      __syncthreads();
      m_blocks = ctx.smem.total_m_blocks[0];
    }
  }

  CUDA_INLINE
  void map_mn_block(uint32_t mn_index, uint32_t &m_id, uint32_t &n_id) {
    if constexpr (kRasterGroupM <= 1 || !kIsDenseGemm) {
      m_id = mn_index / N_BLOCKS;
      n_id = mn_index % N_BLOCKS;
    } else {
      const uint32_t blocks_per_group = kRasterGroupM * N_BLOCKS;
      const uint32_t group_id = mn_index / blocks_per_group;
      const uint32_t first_m = group_id * kRasterGroupM;
      const uint32_t group_m = MIN(m_blocks - first_m, kRasterGroupM);
      const uint32_t idx_in_group = mn_index - group_id * blocks_per_group;
      m_id = first_m + idx_in_group % group_m;
      n_id = idx_in_group / group_m;
    }
  }

  CUDA_INLINE
  bool get_next_block() {
    bool has_next_block = false;
    if (dp_mn_iters) {
      slice_iters = K_BLOCKS;

      map_mn_block(dp_mn_next_index, m_block_id, n_block_id);

      if constexpr (kMultiCastSizeB > 1) {
        m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
      } else if constexpr (kMultiCastSizeA > 1) {
        n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
      }
      k_block_id = 0;
      dp_mn_next_index += gridDim.x / kMultiCastSize;
      dp_mn_iters--;
      has_next_block = true;
    } else if constexpr (kUseStreamK) {
      has_next_block = get_streamk_next_block();
    }

    if constexpr (kIsIndexedGemm) {
      if (has_next_block) fetch_moe_index_block();
    }
    if constexpr (kIsGroupedGemm) {
      if (has_next_block) fetch_moe_group_block();
    }

    return has_next_block;
  };

  CUDA_INLINE
  bool get_streamk_next_block() {
    if (!streamk_mnk_iters) return false;
    uint32_t streamk_mn_index = streamk_mnk_next_index / K_BLOCKS;

    map_mn_block(streamk_mn_index, m_block_id, n_block_id);
    if constexpr (kMultiCastSizeB > 1) {
      m_block_id = m_block_id * kMultiCastSizeB + cluster_rank;
    } else if constexpr (kMultiCastSizeA > 1) {
      n_block_id = n_block_id * kMultiCastSizeA + cluster_rank;
    }
    k_block_id = streamk_mnk_next_index - streamk_mn_index * K_BLOCKS;

    slice_iters = K_BLOCKS - k_block_id;
    slice_iters = slice_iters > streamk_mnk_iters ? streamk_mnk_iters : slice_iters;

    streamk_mnk_iters -= slice_iters;
    streamk_mnk_next_index += slice_iters;

    if (k_block_id == 0) {
      slice_id = 0;
      slice_count = CEIL_DIV(K_BLOCKS - slice_iters, streamk_mnk_total_iters) + 1;
    } else {
      slice_id = k_block_id / streamk_mnk_total_iters;
      uint32_t slice_first_block_iters = k_block_id - slice_id * streamk_mnk_total_iters;
      slice_count = CEIL_DIV(K_BLOCKS - slice_first_block_iters, streamk_mnk_total_iters);
      if (slice_first_block_iters) {
        slice_id++;
        slice_count++;
      }
    }

    slice_id = slice_count - 1 - slice_id;

    locks_offset = streamk_mn_index - dp_mn_total_iters * gridDim.x / kMultiCastSize;
    locks_offset = locks_offset * kMultiCastSize + cluster_rank;

    return true;
  };

  CUDA_INLINE
  void fetch_moe_group_block() {
    uint32_t delta_m_block_id = m_block_id - old_m_block_id;
    offset_in_expert += delta_m_block_id * BlockShape::M;

    while (offset_in_expert >= ctx.smem.expert_tokens[expert_id]) {
      offset_in_expert -= CEIL_DIV(ctx.smem.expert_tokens[expert_id], BlockShape::M) * BlockShape::M;
      expert_id++;
    }

    old_m_block_id = m_block_id;
    if constexpr (Ctx::kIsGroupedMaskedGemm) {
      m_offset = expert_id * expert_max_num_tokens + offset_in_expert;
      current_shape_m = expert_id * expert_max_num_tokens + ctx.smem.expert_tokens[expert_id];
    } else if constexpr (Ctx::kIsGroupedContiguousGemm) {
      m_offset = ctx.smem.expert_offset[expert_id] + offset_in_expert;
      current_shape_m = ctx.smem.expert_offset[expert_id] + ctx.smem.expert_tokens[expert_id];
    }

    if (old_expert_id != expert_id) update_tensor_map_c();
    old_expert_id = expert_id;
  };

  CUDA_INLINE
  void fetch_moe_index_block() {
    if (kUseWarpSpec && ctx.is_math_thread()) return;

    expert_id = ctx.params.expert_ids_ptr[m_block_id];

    const uint32_t *gmem_ptr = ctx.params.sorted_ids_ptr + m_block_id * BlockShape::M;
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(ctx.smem.wr_row_index);

    legacy_load_1d<kUseCpAsync, BlockShape::M / 4, kNumLoadThreads, kLoadThreadOffset>(gmem_ptr_load, smem_ptr_load);
    if constexpr (kUseCpAsync) cp_async_commit_group();
    if constexpr (kUseCpAsync) cp_async_wait_group<0>();

    ctx.sync_load_threads();

    uint32_t thread_id = threadIdx.x;
    if constexpr (kUseWarpSpec) thread_id = thread_id - kNumMathThreads;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
      uint32_t index = kNumLoadThreads * i + thread_id;
      if (index < BlockShape::M) {
        uint32_t idx = ctx.smem.wr_row_index[index];
        ctx.smem.rd_row_index[index] = idx / ctx.params.top_k;
      };
    }

    ctx.sync_load_threads();
  };

  CUDA_INLINE
  void update_tensor_map_c() {
    if constexpr (kIsGroupedGemm && Ctx::kUseTmaC) {
      if (threadIdx.x < 32) {
        __syncwarp();
        if (threadIdx.x == 0) {
          tensor_map_replace_global_dim<1>(ctx.smem.tensor_map_buffer, current_shape_m);
          ctx.params.tensor_map_buffer[blockIdx.x] = ctx.smem.tensor_map_buffer[0];
          tensor_map_release_cta();
          tensor_map_acquire_cta(ctx.params.tensor_map_buffer + blockIdx.x);
        }
        __syncwarp();
      }
    }
  }
};

#include <humming/utils/all.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

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
  using MmaShape = Shape<8, 64, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using DRegisters = float[4];

  CUDA_INLINE
  static void fma(uint64_t &desc, uint32_t *b, float *d, bool pred = true) {
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
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
  static constexpr auto kGemmType = GemmType::GROUPED_CONTIGUOUS;
  static constexpr auto kGemmTypeId = 2u;
  static constexpr auto kIsIndexedGemm = false;
  static constexpr auto kIsGroupedGemm = true;
  static constexpr auto kIsGroupedContiguousGemm = true;
  static constexpr auto kIsGroupedMaskedGemm = false;
};

class TuningConfig {
public:
  static constexpr auto kUseStreamK = false;
  static constexpr auto kNumStages = 4u;
  static constexpr auto kNumCtasPerSm = 1u;
  static constexpr auto kUseWarpSpec = true;
  static constexpr auto kUseMBarrier = true;
  static constexpr auto kUseCpAsync = true;
  static constexpr auto kUseTma = true;
  static constexpr auto kUseTmaA = true;
  static constexpr auto kUseTmaAS = false;
  static constexpr auto kUseTmaB = true;
  static constexpr auto kUseTmaC = false;
  static constexpr auto kUseTmaBS = true;
  static constexpr auto kUseTmaBS2 = false;
  static constexpr auto kUseTmaBZP = false;
  static constexpr auto kUseTmaBias = false;
  static constexpr auto kReduceOverlapLastStageOnly = false;
  static constexpr auto kNumWriteSplits = 1u;
  static constexpr auto kMultiCastSizeA = 1u;
  static constexpr auto kMultiCastSizeB = 1u;
  static constexpr auto kRasterGroupM = 1u;
  static constexpr auto kNumThreads = 640u;
  static constexpr auto kNumMathThreads = 512u;
  static constexpr auto kNumLoadThreads = 128u;
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<8, 256, 128>,
    Shape<8, 32, 64>,
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

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = 8;
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = 256;
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = 128;

extern "C" __constant__ uint32_t WARP_SHAPE_M = 8;
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
extern "C" __constant__ uint32_t GEMM_TYPE_ID = 2;
extern "C" __constant__ uint32_t IS_INDEXED_GEMM = 0;
extern "C" __constant__ uint32_t IS_GROUPED_GEMM = 1;
extern "C" __constant__ uint32_t IS_GROUPED_CONTIGUOUS_GEMM = 1;
extern "C" __constant__ uint32_t IS_GROUPED_MASKED_GEMM = 0;

extern "C" __constant__ uint32_t USE_STREAM_K = 0;
extern "C" __constant__ uint32_t NUM_STAGES = 4;
extern "C" __constant__ uint32_t NUM_CTAS_PER_SM = 1;
extern "C" __constant__ uint32_t USE_WARP_SPEC = 1;
extern "C" __constant__ uint32_t USE_MBARRIER = 1;
extern "C" __constant__ uint32_t USE_CP_ASYNC = 1;
extern "C" __constant__ uint32_t USE_TMA = 1;
extern "C" __constant__ uint32_t USE_TMA_A = 1;
extern "C" __constant__ uint32_t USE_TMA_AS = 0;
extern "C" __constant__ uint32_t USE_TMA_B = 1;
extern "C" __constant__ uint32_t USE_TMA_C = 0;
extern "C" __constant__ uint32_t USE_TMA_BS = 1;
extern "C" __constant__ uint32_t USE_TMA_BS2 = 0;
extern "C" __constant__ uint32_t USE_TMA_BZP = 0;
extern "C" __constant__ uint32_t USE_TMA_BIAS = 0;
extern "C" __constant__ uint32_t REDUCE_OVERLAP_LAST_STAGE_ONLY = 0;
extern "C" __constant__ uint32_t NUM_WRITE_SPLITS = 1;
extern "C" __constant__ uint32_t MULTI_CAST_SIZE_A = 1;
extern "C" __constant__ uint32_t MULTI_CAST_SIZE_B = 1;
extern "C" __constant__ uint32_t RASTER_GROUP_M = 1;
extern "C" __constant__ uint32_t NUM_THREADS = 640;
extern "C" __constant__ uint32_t NUM_MATH_THREADS = 512;
extern "C" __constant__ uint32_t NUM_LOAD_THREADS = 128;
