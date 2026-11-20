// Global 64-row expert groups share the compact activation workspace in windows.
constexpr int WINDOW_JOBS = 48;
__device__ __forceinline__ bool select_window(Params& p) {
  if (!p.layout) return true;
  int start = p.layout[1] + p.window * WINDOW_JOBS;
  p.jobs = min(WINDOW_JOBS, p.layout[0] / 64 - start);
  if (p.jobs <= 0) return false;
  p.sorted += size_t(start) * 64;
  p.expert_map += start;
  return true;
}
__device__ __forceinline__ int source_slot(const Params& p,const Smem& s,int index) {
  return p.layout ? p.sorted[index] : int(s.ent[index]);
}
__device__ void route_window(const Params& p,Smem& s) {
  if (!p.layout) { route(p,s); return; }
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  for (int j=warp;j<p.jobs;j+=NT/32) {
    const int* slots=p.sorted+j*64;
    const int count=__popc(__ballot_sync(0xffffffffu,slots[lane]<p.M*8))+
                    __popc(__ballot_sync(0xffffffffu,slots[lane+32]<p.M*8));
    if (!lane) { s.jC[j]=count; s.jE[j]=p.expert_map[j]; s.jS[j]=j*64; }
  }
  if (!tid) { s.njF=p.jobs; s.njS=0; }
  asm volatile("bar.sync 1, %0;" :: "n"(NT) : "memory");
}

