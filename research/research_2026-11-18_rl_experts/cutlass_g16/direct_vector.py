"""Aligned 4-lane vector producer for one native Humming shared-memory 64x64 weight tile."""
VECTOR_BODY=r"""
    static_assert(!Ctx::kUseWarpSpec && !Ctx::kUseMBarrier && !kUseTma);
    static_assert(ElementA::kBits==16 && ElementB::kBits==4 && !Ctx::kUsePackedKLayout);
    static_assert(BlockShape::N%64==0 && BlockShape::K%64==0 && kNumLoadThreads%32==0);
    constexpr uint32_t N=ProblemShape::N,K=ProblemShape::K,tile_n=BlockShape::N,tile_k=BlockShape::K;
    const auto* q=reinterpret_cast<uint8_t const*>(gmem_ptr_raw);
    const uint64_t first=uint64_t(gmem_ptr-gmem_ptr_raw)*4;
    const uint64_t erow=first/(N*2);
    const uint64_t e=erow/(K/16);
    const uint32_t first_k=uint32_t(erow%(K/16))*16;
    const uint32_t first_n=uint32_t(first%(N*2))/2;
    const uint32_t tid=ctx.load_thread_id(),lane=tid%32;
    const uint32_t r=lane/4,k16=lane%4;
    for (uint32_t tile=tid/32;tile<(tile_n/64)*(tile_k/64);tile+=kNumLoadThreads/32) {
      const uint32_t tn=tile%(tile_n/64),tk=tile/(tile_n/64);
      const uint32_t srcn=first_n+tn*64+r;
      const uint32_t srck=(first_k+tk*64+k16*16)/2;
      uint64_t lo[4],hi[4];
      PRAGMA_UNROLL
      for (int a=0;a<4;++a) {
        const auto* row=q+(e*N+srcn+a*16)*(K/2)+srck;
        lo[a]=*reinterpret_cast<uint64_t const*>(row);
        hi[a]=*reinterpret_cast<uint64_t const*>(row+8*(K/2));
      }
      PRAGMA_UNROLL
      for (int p=0;p<4;++p) {
        uint32_t packed[4];
        PRAGMA_UNROLL
        for (int a=0;a<4;++a) {
          const uint32_t b0=(lo[a]>>(8*p))&255u;
          const uint32_t b3=(hi[a]>>(8*p+32))&255u;
          const uint32_t upper=(lo[a]>>(8*p+32))&255u;
          const uint32_t lower=(hi[a]>>(8*p))&255u;
          constexpr bool wg=REPLACE_WGMMA;
          const uint32_t b1=wg?lower:upper,b2=wg?upper:lower;
          const uint32_t l=(b0&15)|((b1&15)<<4)|((b2&15)<<8)|((b3&15)<<12);
          const uint32_t h=((b0&240)>>4)|(b1&240)|((b2&240)<<4)|((b3&240)<<8);
          packed[a]=l|(h<<16);
        }
        const uint32_t native=(tk*4+k16)*(2*tile_n)+tn*128+(r*4+p)*4;
        reinterpret_cast<uint4*>(smem_ptr)[native/4]=uint4{packed[0],packed[1],packed[2],packed[3]};
      }
    }
"""
