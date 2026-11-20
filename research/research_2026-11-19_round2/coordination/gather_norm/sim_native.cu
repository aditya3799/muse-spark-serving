/**
 * \brief Fused post-norm + gated residual + pre-norm for the MuseSpark v1.2 layer boundary.
 *
 * branch: [M, H] bf16, the sublayer output.
 * stream: [M, H] fp32, the residual stream.
 * output: [M, H] bf16, pre-norm of the new stream (next sublayer input).
 * new_stream: [M, H] fp32, gate_a * stream + gate_b * post_norm(branch).
 * router_out: [M, H] fp32, fp32 copy of output; null when unused.
 * post_weight / pre_weight: [H] bf16, null when absent.
 * alpha / beta: [H] or [1] fp32 residual gates, null when ungated.
 */

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/cta.cuh>
#include <sgl_kernel/distributed/ptx.cuh>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/mbarrier.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cooperative_groups.h>
#include <cstdint>

namespace sglang {

struct NormResidualParams {
  const void* __restrict__ branch;
  const void* __restrict__ stream;
  void* __restrict__ output;
  void* __restrict__ new_stream;
  void* __restrict__ router_out;
  const void* __restrict__ post_weight;
  const void* __restrict__ pre_weight;
  const void* __restrict__ alpha;
  const void* __restrict__ beta;
  float post_eps;
  float pre_eps;
  int num_rows;
  bool gate_channel_wise;
  bool has_pre_weight;
  bool pre_norm_bf16;
};

namespace device::norm_residual {

// 16B per vectorized bf16 access; the fp32 tensors are covered by 256-bit
// accesses over the same element range (two 128-bit halves pre-Blackwell) so
// both dtypes share one thread map. All row data lives in registers as
// fp32x2_t pairs, so conversions and the bf16 round-trip use the paired
// cvt forms instead of per-element scalars.
inline constexpr int kVec = 8;
inline constexpr int kFVec = kMaxVecBytes >= 32 ? kVec : kVec / 2;
inline constexpr int kFParts = kVec / kFVec;
inline constexpr int kPairs = kVec / 2;
inline constexpr int kFPairs = kFVec / 2;

// Triton's `.to(tl.bfloat16).to(tl.float32)` round-trip; load-bearing for parity.
SGL_DEVICE fp32x2_t round_bf16(fp32x2_t v) {
  return cast<fp32x2_t>(cast<bf16x2_t>(v));
}

}  // namespace device::norm_residual

template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    muse_spark_norm_residual_kernel(const __grid_constant__ NormResidualParams params) {
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
namespace muse_ag_norm {
inline constexpr uint32_t kVec=8;
SGL_DEVICE fp32x2_t mul_pair(fp32x2_t a, fp32x2_t b) {
  return {a.x * b.x, a.y * b.y};
}

SGL_DEVICE fp32x2_t fma_pair(fp32x2_t a, fp32x2_t b, fp32x2_t c) {
  return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
}

SGL_DEVICE bool vector_pending(uint4 raw) {
  return (raw.x & 0xffffu) == 0 || (raw.x >> 16) == 0 || (raw.y & 0xffffu) == 0 || (raw.y >> 16) == 0 ||
         (raw.z & 0xffffu) == 0 || (raw.z >> 16) == 0 || (raw.w & 0xffffu) == 0 || (raw.w >> 16) == 0;
}

// Spins until every rank's copy of the vector at (row, col) arrived, then clears the slot.
template <uint32_t kNLocal>
SGL_DEVICE uint4 take_gathered(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col) {
  bf16_t* src = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
  uint4 raw;
  bool pending;
  do {
    device::ptx::ld_relaxed_16B(raw, src, 0);
    pending = vector_pending(raw);
  } while (pending);
  device::AlignedVector<uint32_t, 4> zero;
  zero.fill(0);
  zero.store(src);
  return raw;
}

template <int kPacks, int kVectorStride, uint32_t kNLocal>
SGL_DEVICE uint4
receive_pack(bf16_t* gathered, uint32_t num_rows, uint32_t row, uint32_t col, int pack, uint4 (&cache)[2]) {
  if (pack % 2 == 0) {
    if (pack + 1 < kPacks) {
      const uint32_t next_col = col + kVectorStride * kVec;
      bf16_t* src0 = gathered + ((col / kNLocal) * num_rows + row) * kNLocal + col % kNLocal;
      bf16_t* src1 = gathered + ((next_col / kNLocal) * num_rows + row) * kNLocal + next_col % kNLocal;
      bool pending;
      do {
        device::ptx::ld_relaxed_16B(cache[0], src0, 0);
        device::ptx::ld_relaxed_16B(cache[1], src1, 0);
        pending = vector_pending(cache[0]) | vector_pending(cache[1]);
      } while (pending);
      device::AlignedVector<uint32_t, 4> zero;
      zero.fill(0);
      zero.store(src0);
      zero.store(src1);
    } else {
      cache[0] = take_gathered<kNLocal>(gathered, num_rows, row, col);
    }
  }
  return cache[pack % 2];
}


}
struct SimParams { void* gathered; uint32_t num_rows; NormResidualParams norm; };
template <
    int64_t kHidden,
    uint32_t kWorld,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    sim_gather_norm_kernel(const __grid_constant__ SimParams params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  constexpr uint32_t kNLocal = static_cast<uint32_t>(kHidden) / kWorld;
  static_assert(kHidden % (kNumThreads * kVec) == 0);
  static_assert(kNLocal % kVec == 0, "a vector must never cross a rank block");

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const auto& norm = params.norm;
  const auto post_weight = static_cast<const bf16_t*>(norm.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(norm.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(norm.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(norm.beta);

  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && norm.gate_channel_wise;
  const bool has_pre_weight = norm.has_pre_weight;
  const bool pre_bf16 = norm.pre_norm_bf16;

  const auto gathered = reinterpret_cast<bf16_t*>(params.gathered);
  for (uint32_t row = blockIdx.x; row < params.num_rows; row += gridDim.x) {
    const int64_t base = static_cast<int64_t>(row) * kHidden;
    const auto stream = static_cast<const fp32_t*>(norm.stream) + base;
    const auto out = static_cast<bf16_t*>(norm.output) + base;
    const auto new_stream = static_cast<fp32_t*>(norm.new_stream) + base;

    fp32x2_t vals[kPacks][kPairs];
    fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
    F32Vec stream_v[kPacks][kFParts];
    F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
    BF16Vec pre_w_pf[kPreSlots];
    uint4 receive_cache[2];

    float sum_sq = 0.0f;
#pragma unroll
    for (int p = 0; p < kPacks; ++p) {
      const int idx = threadIdx.x + p * kNumThreads;
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
      const uint4 raw = muse_ag_norm::receive_pack<kPacks, kNumThreads, kNLocal>(
          gathered, params.num_rows, row, idx * kVec, p, receive_cache);
      const auto& branch_v = *reinterpret_cast<const BF16Vec*>(&raw);
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        auto v = cast<fp32x2_t>(branch_v[q]);
        if constexpr (kPostNormIsInputScale && kPostWeight) {
          v = muse_ag_norm::mul_pair(v, post_w[p][q]);
          v = round_bf16(v);
        }
        vals[p][q] = v;
        sum_sq = fmaf(v.x, v.x, sum_sq);
        sum_sq = fmaf(v.y, v.y, sum_sq);
      }
    }

    sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
    const float post_rstd = math::rsqrt(sum_sq / kHidden + norm.post_eps);

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
        normed_b = muse_ag_norm::mul_pair(normed_b, {post_rstd, post_rstd});
        if constexpr (!kPostNormIsInputScale && kPostWeight) {
          normed_b = muse_ag_norm::mul_pair(normed_b, post_w[p][q]);
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
    const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + norm.pre_eps);

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
#pragma unroll
      for (int q = 0; q < kPairs; ++q) {
        fp32x2_t v = vals[p][q];
        v = muse_ag_norm::mul_pair(v, {pre_rstd, pre_rstd});
        if (has_pre_weight) {
          const auto wf = cast<fp32x2_t>(weight_v[q]);
          v = muse_ag_norm::mul_pair(v, wf);
        }
        if (pre_bf16) v = round_bf16(v);
        out_v[q] = cast<bf16x2_t>(v);
      }
      out_v.store(out, idx);
    }
  }

  __syncthreads();
}
}

using namespace sglang;
__global__ void sim_pack(const __nv_bfloat16* in, __nv_bfloat16* packed, int rows) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=rows*8192)return;
 int row=i/8192,col=i%8192;uint16_t word=reinterpret_cast<const uint16_t*>(in)[i];
 if(word==0)word=0x8000;
 reinterpret_cast<uint16_t*>(packed)[(col/1024)*rows*1024+row*1024+col%1024]=word;
}
__global__ void sim_gather(__nv_bfloat16* packed,__nv_bfloat16* out,int rows) {
 using namespace sglang::device;
 constexpr int N=8192,kNLocal=1024,kVec=8,kBlock=128;
 using vec_t=AlignedVector<bf16x2_t,4>;
 int tid=blockIdx.x*kBlock+threadIdx.x;
 int stride=gridDim.x*kBlock*kVec;
 for(int elem=tid*kVec;elem<rows*N;elem+=stride){
  int row=elem/N,col=elem%N;
  auto src=packed+((col/kNLocal)*rows+row)*kNLocal+col%kNLocal;
  uint4 raw;
  do {ptx::ld_relaxed_16B(raw,src,0);} while(sglang::muse_ag_norm::vector_pending(raw));
  const auto& gathered=*reinterpret_cast<const vec_t*>(&raw);
  vec_t out_vec;
  #pragma unroll
  for(int j=0;j<4;++j)out_vec[j]=gathered[j];
  out_vec.store(out+elem);
  AlignedVector<uint32_t,4> zero;zero.fill(0);zero.store(src);
 }
}
template<int T,bool S,bool G> int invoke(int candidate,void* packed,const NormResidualParams& p,int cap,cudaStream_t stream) {
 if(candidate)sim_gather_norm_kernel<8192,8,T,false,(T==256),S,true,G><<<(p.num_rows<cap?p.num_rows:cap),T,0,stream>>>(SimParams{packed,uint32_t(p.num_rows),p});
 else muse_spark_norm_residual_kernel<8192,T,false,(T==256),S,false,true,G><<<p.num_rows,T,0,stream>>>(p);
 return int(cudaGetLastError());
}
extern "C" size_t norm_params_size(){return sizeof(NormResidualParams);}
extern "C" size_t norm_offset(int i) {
 static const size_t offsets[]={offsetof(NormResidualParams,branch),offsetof(NormResidualParams,stream),offsetof(NormResidualParams,output),offsetof(NormResidualParams,new_stream),offsetof(NormResidualParams,router_out),offsetof(NormResidualParams,post_weight),offsetof(NormResidualParams,pre_weight),offsetof(NormResidualParams,alpha),offsetof(NormResidualParams,beta),offsetof(NormResidualParams,post_eps),offsetof(NormResidualParams,pre_eps),offsetof(NormResidualParams,num_rows),offsetof(NormResidualParams,gate_channel_wise),offsetof(NormResidualParams,has_pre_weight),offsetof(NormResidualParams,pre_norm_bf16)};
 return i>=0 && i<15 ? offsets[i] : size_t(-1);
}
extern "C" int pack_sim(const void* in,void* packed,int rows,cudaStream_t stream) {
 if(!in||!packed||rows<1)return int(cudaErrorInvalidValue);
 sim_pack<<<(rows*8192+255)/256,256,0,stream>>>(static_cast<const __nv_bfloat16*>(in),static_cast<__nv_bfloat16*>(packed),rows);return int(cudaGetLastError());
}
extern "C" int gather_sim(void* packed,void* out,int rows,int cap,cudaStream_t stream) {
 int blocks=(rows*8192/8+127)/128;blocks=blocks<cap-1?blocks:cap-1;sim_gather<<<blocks,128,0,stream>>>(static_cast<__nv_bfloat16*>(packed),static_cast<__nv_bfloat16*>(out),rows);return int(cudaGetLastError());
}
extern "C" int norm_sim(int candidate,int scale,int gate,int threads,int cap,void* packed,const void* raw,cudaStream_t stream) {
 if(!raw||!packed||(threads!=256&&threads!=512))return int(cudaErrorInvalidValue);
 const auto& p=*static_cast<const NormResidualParams*>(raw);
 if(p.num_rows<1||!p.post_weight||!p.stream||!p.output||!p.new_stream)return int(cudaErrorInvalidValue);
 #define CHOOSE(T) if(scale){if(gate)return invoke<T,true,true>(candidate,packed,p,cap,stream);return invoke<T,true,false>(candidate,packed,p,cap,stream);}else{if(gate)return invoke<T,false,true>(candidate,packed,p,cap,stream);return invoke<T,false,false>(candidate,packed,p,cap,stream);}
 if(threads==256){CHOOSE(256)}else{CHOOSE(512)}
 #undef CHOOSE
}
