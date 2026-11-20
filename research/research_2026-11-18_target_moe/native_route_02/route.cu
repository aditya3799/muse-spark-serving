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
__device__ __forceinline__ float serving_scaled(float dot, float scale) {
  float rounded = __bfloat162float(__float2bfloat16_rn(dot));
  return __bfloat162float(__float2bfloat16_rn(rounded * scale));
}
__device__ __forceinline__ __nv_bfloat16 serving_activation(float gate, float up, float scale) {
  float g = serving_scaled(gate, scale);
  float u = serving_scaled(up, scale);
  return __float2bfloat16_rn((g / (1.f + expf(-g))) * u);
}
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
                     // OKC_FTSC2; the TINY builds measured -0.5%, so off by default)
namespace okc {
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
constexpr int HD = 4096, ID = 4096;
constexpr int NW = 16, NT = NW * 32;
constexpr int NTL = NT + (1 ? 64 : 0 ? 32 : 0); // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = 0 ? 4 : 2; // cross-warp partial-sum buffers
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
constexpr int NCH = 64 / NW; // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
constexpr int SK_T = 8; // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = 8 > 8 ? 8 : 4;
static_assert(!0 || 8 == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = 64; // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16; // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30; // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * 4; return (1 && (!0 || 0) && 1 && 1 && 8 == 8 && !0 && !0) ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(3 <= 7 && 3 <= 5, "barrier slots"); // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8; // skinny x row pitch (bf16)
constexpr int KS = 256; // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8; // fat x row pitch (bf16)
constexpr int NST = 3; // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;
struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw; // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf; // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu; // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn; // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs; // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB; // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready; // WGS: per job, rows of xA written
  int* doneAs; // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero; // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr; // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA; // [JMAX]
  __nv_bfloat16* inter; // [EMAX, ID]
  float* ybuf; // [EMAX, HD]
  float* ss; // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1; // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
};
struct Smem {
  uint64_t bars[24];
  int wcp[JMAX + 1], wcpB[JMAX + 1], jrow[JMAX + 1];
  uint32_t wtot, wtotB;
  int wgr[4], wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], njF, njS, gen;
  signed char eid[EMAX];
  short ent[EMAX];
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = __shfl_sync(0xffffffffu, (tid) >> 5, 0);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
  { // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
  asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;


    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
  }
  asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");

  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;

    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
}
__device__ void early_issue(const Params&, Smem&) {}
__device__ __forceinline__ int wgs_kperm3_col(int col) {
  return (col & ~63) + (((col >> 1) & 3) << 4) + (((col >> 3) & 1) << 3)
         + (((col >> 4) & 3) << 1) + (col & 1);
}
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const int pcol = wgs_kperm3_col(col);
    const float gsc = p.gsA[e];
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (pcol & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const auto v = serving_activation(d4[h], d4[2+h], gsc);
        st_keep_bf16(xb + (size_t)((((pcol >> 8) * ntt + (tk >> 3)) * 32 + ((pcol & 255) >> 3)) * 64 + (tk & 7) * 8),
                     v);
      }
    }
  } else {
    const int col = r16 * 16 + g;
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        auto* dst = p.out + (size_t)S.ent[jst + tk] * HD + col;
        st_keep_bf16(dst, __float2bfloat16_rn(d4[h]));
        st_keep_bf16(dst + 8, __float2bfloat16_rn(d4[2+h]));
      }
    }
  }
}

__global__ void route_fixture(Params p) {
  extern __shared__ __align__(16) unsigned char buffer[];
  auto& S = *reinterpret_cast<Smem*>(buffer);
  route(p,S);
  int lane = threadIdx.x & 31;
  int r16 = blockIdx.x * NW + threadIdx.x / 32;
  int col = r16 * 16 + lane / 4;
  for (int j = 0; j < S.njF + S.njS; ++j) {
    for (int nt = 0; nt < (S.jC[j]+7)/8; ++nt) {
      float d[4] = {};
      for (int h = 0; h < 2; ++h) {
        int k = nt * 8 + (lane % 4) * 2 + h;
        if (k < S.jC[j]) {
          int slot = S.ent[S.jS[j]+k];
          int e = S.jE[j];
          const auto* x = p.x + size_t(slot >> 3) * HD + col;
          float offset = (e+1) * 0.03125f + (slot & 7) * 0.125f;
          d[h] = __bfloat162float(x[0]) * ((e & 3)+1) + offset;
          d[2+h] = __bfloat162float(x[8]) * ((e & 3)+1) + offset;
        }
      }
      wgs_epi_tile<true>(p,S,j,r16,nt,d);
    }
  }
}
}
extern "C" int expert_run(const void*, const void*, const void*, const void*, const __nv_bfloat16* x,
                          const int* ids,__nv_bfloat16* out,void*,void*,void*,int m,int e,int,cudaStream_t stream) {
  okc::Params p{};
  p.M=m; p.x=x; p.topk_ids=ids; p.out=out; p.nloc=e;
  okc::route_fixture<<<16,okc::NT,sizeof(okc::Smem),stream>>>(p);
  return cudaGetLastError();
}
