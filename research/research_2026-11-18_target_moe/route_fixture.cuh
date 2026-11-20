
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
