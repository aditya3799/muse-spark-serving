"""Build a private upstream Humming kernel with direct canonical-word producers for the legacy pipeline."""
from pathlib import Path
from functools import lru_cache
import hashlib

W_BODY=r"""
    static_assert(!Ctx::kUseWarpSpec && !Ctx::kUseMBarrier && !kUseTma);
    static_assert(ElementA::kBits==16 && ElementB::kBits==4 && !Ctx::kUsePackedKLayout);
    const auto *q=reinterpret_cast<uint8_t const*>(gmem_ptr_raw);
    auto *native=reinterpret_cast<uint32_t*>(smem_ptr);
    const uint64_t first=uint64_t(gmem_ptr-gmem_ptr_raw)*4;
    for (uint32_t u=ctx.load_thread_id();u<kNumInt4s*4;u+=kNumLoadThreads) {
      constexpr uint32_t N=ProblemShape::N, K=ProblemShape::K;
      const uint64_t flat=first+uint64_t(u/(BlockShape::N*2))*(N*2)+(u%(BlockShape::N*2));
      const uint32_t v=flat%128;
      const uint32_t lane=v/4;
      const uint32_t n0=(flat%(N*2))/128*64+(v%4)*16+lane/4;
      const uint64_t row=flat/(N*2);
      const uint32_t kbase=(row%(K/16))*8+lane%4;
      const uint64_t e=row/(K/16);
      uint32_t word=0;
      PRAGMA_UNROLL
      for (int j=0;j<4;++j) {
        constexpr bool wg=REPLACE_WGMMA;
        const uint32_t nh=wg?(j%2):(j/2);
        const uint32_t kh=wg?(j/2):(j%2);
        const uint32_t b=q[(e*N+n0+nh*8)*(K/2)+kbase+kh*4];
        word|=((b&15)<<(j*4))|((b&240)<<(j*4+12));
      }
      native[u]=word;
    }
"""
S_BODY=r"""
    static_assert(!Ctx::kUseWarpSpec && !Ctx::kUseMBarrier && !kUseTma);
    static_assert(kGroupSize==16 && ElementBS::kBits==8 && !kUseMxScale && !kIsBlock && !kIsChannel);
    const auto *s=reinterpret_cast<uint32_t const*>(gmem_ptr_raw);
    auto *native=reinterpret_cast<uint32_t*>(smem_ptr);
    const uint64_t first=uint64_t(gmem_ptr-gmem_ptr_raw)*16;
    for (uint32_t u=ctx.load_thread_id();u<kNumInt4s*4;u+=kNumLoadThreads) {
      constexpr uint32_t N=ProblemShape::N, K=ProblemShape::K;
      const uint64_t flat=first+uint64_t(u/(BlockShape::N/4))*N+(u%(BlockShape::N/4))*4;
      const uint64_t row=flat/N;
      const uint64_t e=row/(K/16);
      const uint32_t kgroup=row%(K/16);
      uint32_t word=0;
      PRAGMA_UNROLL
      for (uint32_t j=0;j<4;++j) {
        const uint32_t n=(flat%N)+j;
        const uint32_t logical=n/64*64+(n%8)*8+(n%64)/8;
        const uint32_t scales=s[(e*(K/64)+kgroup/4)*N+logical];
        word|=((scales>>((kgroup%4)*8))&255u)<<(8*j);
      }
      native[u]=word;
    }
"""

@lru_cache(maxsize=8)
def adapted_source(wgmma,vector=False,scales_only=False):
    import humming
    from direct_vector import VECTOR_BODY
    weight_body=VECTOR_BODY if vector else W_BODY
    headers=Path(humming.__file__).resolve().parent/"include/humming"
    expected={"kernel/humming.cuh":"9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3",
              "memory/g2s_pipeline.cuh":"479ff827ecf40b82d4685b3a104d796ef41bb643a1d9cc398feabcc77ccbd43b",
              "memory/g2s_loader/loader_b.cuh":"f2dcc1925193244bf6ac9cffb56aa2881fb602d92ef9ba710460e9a7f09d550b",
              "memory/g2s_loader/loader_bs.cuh":"a03b01ceb6a9e1b32711d8e05aa0998df430ecc640299024a9ab60e215e12e53"}
    if any(hashlib.sha256((headers/file).read_text().encode()).hexdigest()!=digest for file,digest in expected.items()):
        return None
    pipe=(headers/"memory/g2s_pipeline.cuh").read_text()
    for name,body in (("b",weight_body.replace("REPLACE_WGMMA","true" if wgmma else "false")),("bs",S_BODY)):
        if scales_only and name=="b":
            continue
        relative=f"memory/g2s_loader/loader_{name}.cuh"
        code=(headers/relative).read_text()
        a=code.index("  void load_legacy(int4 *smem_ptr) {")+len("  void load_legacy(int4 *smem_ptr) {")
        b=code.index("\n  }\n\n  CUDA_INLINE\n  void advance()",a)
        old="#include <humming/"+relative+">"
        assert pipe.count(old)==1
        pipe=pipe.replace(old,code[:a]+body.rstrip()+code[b:])
    kernel=(headers/"kernel/humming.cuh").read_text()
    old="#include <humming/memory/g2s_pipeline.cuh>"
    assert kernel.count(old)==1
    return kernel.replace(old,pipe)
