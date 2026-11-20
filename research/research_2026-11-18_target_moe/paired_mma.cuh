// Test adapter: execute the logical Hopper fragments using SM86 warp matrix instructions.
__device__ void index_jobs(const Params& p, Smem& s) {
  route(p,s);
  if (threadIdx.x == 0) {
    int row = 0;
    for (int j = 0; j < s.njF+s.njS; ++j) {
      s.jrow[j] = row;
      row += (s.jC[j]+7) & ~7;
    }
    s.jrow[s.njF+s.njS] = row;
  }
  __syncthreads();
}
__global__ void gather_phase(Params p) {
  auto& s = *reinterpret_cast<Smem*>(smem_raw);
  index_jobs(p,s);
  wgs_gather(p,s);
}

template<bool DOWN>
__global__ void project_phase(Params p) {
  auto& s = *reinterpret_cast<Smem*>(smem_raw);
  index_jobs(p,s);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r16 = blockIdx.x*16 + warp;
  const auto* bank = DOWN ? p.wBs : p.wAs;
  const auto* table = reinterpret_cast<const uint16_t*>(p.lut);
  for (int j = 0; j < s.njF + s.njS; ++j) {
    const int tiles = (s.jC[j]+7)/8;
    const auto* activations = (DOWN ? p.xB : p.xA) + size_t(s.jrow[j])*HD;
    const int unit = s.jE[j]*(DOWN ? 16 : 32) + blockIdx.x;
    for (int nt = 0; nt < tiles; ++nt) {
      float d[4] = {}, odd[4] = {};
      for (int st = 0; st < 16; ++st) {
        const auto* piece = bank + ((size_t(unit)*16+st)*4 + warp/4)*9216;
        for (int group = 0; group < 4; ++group) {
          const int i = (group*4+warp%4)*32 + lane;
          const unsigned sc = reinterpret_cast<const uint16_t*>(piece+8192)[i];
#pragma unroll
          for (int step = 0; step < 4; ++step) {
            const unsigned q = reinterpret_cast<const unsigned*>(piece)[i*4+step];
            const unsigned a0=serving_pair<0>(q,sc,table), a1=serving_pair<1>(q,sc,table);
            const unsigned a2=serving_pair<2>(q,sc,table), a3=serving_pair<3>(q,sc,table);
            const int k8=(group*4+step)*2;
            const auto* base=activations+(((st*tiles+nt)*32+k8)*64+(lane/4)*8+(lane%4)*2);
            const unsigned b0=*reinterpret_cast<const unsigned*>(base);
            const unsigned b1=*reinterpret_cast<const unsigned*>(base+64);
            float* f=(tiles==1 && step>=2) ? odd : d;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(f[0]),"+f"(f[1]),"+f"(f[2]),"+f"(f[3])
                         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
          }
        }
      }
      if (tiles==1) for (int k=0;k<4;++k) d[k]+=odd[k];
      wgs_epi_tile<DOWN>(p,s,j,r16,nt,d);
    }
  }
}
}
extern "C" int paired(const unsigned char* a,const unsigned char* b,const float* scale,
                      const uint32_t* table,const __nv_bfloat16* x,const int* ids,
                      __nv_bfloat16* out,__nv_bfloat16* tiles,int* ready,int m,int experts,cudaStream_t stream) {
  using namespace okc;
  Params p{}; p.wAs=a; p.wBs=b; p.gsA=scale; p.lut=table;
  p.x=x; p.topk_ids=ids; p.out=out; p.M=m; p.nloc=experts; p.xready=ready;
  p.xA=tiles; p.xB=tiles+size_t(EMAX+8*JMAX)*HD;
  gather_phase<<<16,NT,sizeof(Smem),stream>>>(p);
  project_phase<false><<<32,NT,sizeof(Smem),stream>>>(p);
  project_phase<true><<<16,NT,sizeof(Smem),stream>>>(p);
  return cudaGetLastError();
}
