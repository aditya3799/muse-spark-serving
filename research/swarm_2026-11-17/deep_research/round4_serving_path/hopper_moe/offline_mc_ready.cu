#include "mc_ready_plane_handoff.cuh"
#include "norm_residual.cuh"
using namespace sglang;
extern "C" __global__ __launch_bounds__(1024,1) void stock_mc_reduce(const AllReducePullParams<8> params) {
 using T=bf16_t;constexpr uint32_t kWorldSize=8;constexpr bool kUsePDL=false;
 using V=device::AlignedVector<bf16x2_t,4>;using Impl=MultiCastImpl<V,8,false>;

  using namespace device;
  constexpr uint32_t kVecSize = 16 / (sizeof(T) * 2);
  using vec_t = AlignedVector<packed_t<T>, kVecSize>;
  const auto num_total_vecs = params.num_vecs;
  const auto avg_vecs = num_total_vecs / kWorldSize;
  const auto rem_vecs = num_total_vecs % kWorldSize;
  const auto num_vecs = avg_vecs + (params.rank < rem_vecs ? 1 : 0);
  const auto vec_offset = params.rank * avg_vecs + min(params.rank, rem_vecs);
  const auto impl = Impl{params, vec_offset};
  PDLWaitPrimary<kUsePDL>();
  const auto barrier = distributed::Barrier<kWorldSize>{
      params.ws.semaphores.data(),
      params.rank,
      /*num_arrives=*/2,
  };
  barrier.arrive_relaxed(/*n=*/0);
  __syncthreads();

  const auto num_threads = blockDim.x * gridDim.x;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    impl.load_reduce(vec, vid);
    impl.store_multi(vec, vid);
  }

  PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  barrier.arrive_rel_acq(/*n=*/1);
}
extern "C" __global__ __launch_bounds__(256) void stock_norm_small(const NormResidualParams params) {
 constexpr int64_t kHidden=8192;constexpr int kNumThreads=256;constexpr bool kUsePDL=false,kPrefetchWeights=true,kPostNormIsInputScale=false,kHasRouter=true,kPostWeight=true,kGate=true;

  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  static_assert(kHidden % (kNumThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);

  // [0, kNumWarps) and [kNumWarps, 2*kNumWarps) are the two reduction scratch
  // areas; separate areas keep the second reduction free of a WAR barrier.
  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;
  const bool pre_bf16 = params.pre_norm_bf16;

  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  F32Vec stream_v[kPacks][kFParts];
  F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      stream_v[p][j].load(stream, idx * kFParts + j);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
            beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    if constexpr (kPostWeight) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q)
        post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale && kPostWeight) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kHidden + params.post_eps);

  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          if constexpr (kPrefetchWeights) {
            alpha_v[j] = alpha_pf[p][j];
            beta_v[j] = beta_pf[p][j];
          } else {
            alpha_v[j].load(alpha_ptr, idx * kFParts + j);
            beta_v[j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    F32Vec new_stream_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      new_stream_v[h][l] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[p][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      new_stream_v[j].store(new_stream, idx * kFParts + j);
  }

  pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + params.pre_eps);

#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec weight_v;
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights) {
        weight_v = pre_w_pf[p];
      } else {
        weight_v.load(pre_weight, idx);
      }
    }
    BF16Vec out_v;
    F32Vec router_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      if (pre_bf16) v = round_bf16(v);
      out_v[q] = cast<bf16x2_t>(v);
      if constexpr (kHasRouter) {
        router_v[q / kFPairs][q % kFPairs] = v;
      }
    }
    out_v.store(out, idx);
    if constexpr (kHasRouter) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        router_v[j].store(router, idx * kFParts + j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}
extern "C" __global__ __launch_bounds__(512) void stock_norm_large(const NormResidualParams params) {
 constexpr int64_t kHidden=8192;constexpr int kNumThreads=512;constexpr bool kUsePDL=false,kPrefetchWeights=false,kPostNormIsInputScale=false,kHasRouter=true,kPostWeight=true,kGate=true;

  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  static_assert(kHidden % (kNumThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);

  // [0, kNumWarps) and [kNumWarps, 2*kNumWarps) are the two reduction scratch
  // areas; separate areas keep the second reduction free of a WAR barrier.
  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;
  const bool pre_bf16 = params.pre_norm_bf16;

  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  F32Vec stream_v[kPacks][kFParts];
  F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
  BF16Vec pre_w_pf[kPreSlots];

  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      stream_v[p][j].load(stream, idx * kFParts + j);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
            beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    if constexpr (kPostWeight) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q)
        post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale && kPostWeight) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kHidden + params.post_eps);

  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          if constexpr (kPrefetchWeights) {
            alpha_v[j] = alpha_pf[p][j];
            beta_v[j] = beta_pf[p][j];
          } else {
            alpha_v[j].load(alpha_ptr, idx * kFParts + j);
            beta_v[j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    F32Vec new_stream_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      new_stream_v[h][l] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[p][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      new_stream_v[j].store(new_stream, idx * kFParts + j);
  }

  pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + params.pre_eps);

#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec weight_v;
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights) {
        weight_v = pre_w_pf[p];
      } else {
        weight_v.load(pre_weight, idx);
      }
    }
    BF16Vec out_v;
    F32Vec router_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      if (pre_bf16) v = round_bf16(v);
      out_v[q] = cast<bf16x2_t>(v);
      if constexpr (kHasRouter) {
        router_v[q / kFPairs][q % kFPairs] = v;
      }
    }
    out_v.store(out, idx);
    if constexpr (kHasRouter) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        router_v[j].store(router, idx * kFParts + j);
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}
