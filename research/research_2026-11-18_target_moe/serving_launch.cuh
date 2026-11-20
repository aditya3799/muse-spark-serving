
extern "C" void expert_sizes(size_t* counts) {
  using namespace okc;
  counts[0] = 8 + 512 + JMAX;
  counts[1] = JMAX * 16;
  counts[2] = 2ull * WGS_XROWS * HD;
}

extern "C" int expert_config(int* grid) {
  using namespace okc;
  int dev, coop, sms, active;
  constexpr size_t shared = ((sizeof(Smem) + 127) / 128) * 128 +
      (size_t)WGS_NS * UNIT_BYTES + (size_t)WGS_NX * WGS_XS;
#define CHECK_CUDA(call) do { auto err = (call); if (err != cudaSuccess) return int(err); } while (0)
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
  if (!coop) return int(cudaErrorNotSupported);
  CHECK_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK_CUDA(cudaFuncSetAttribute(moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, moe_fused<false>, NTL, shared));
  if (active < 1) return int(cudaErrorLaunchOutOfResources);
  *grid = sms < WGS_SLOTS ? sms : WGS_SLOTS - 1;
  return 0;
}

extern "C" int expert_run(const unsigned char* w13, const unsigned char* w2, const float* scale,
                          const uint16_t* table, const __nv_bfloat16* x, const int* ids,
                          __nv_bfloat16* out, int* counters, float* fix, __nv_bfloat16* tiles,
                          int rows, int experts, int grid, cudaStream_t stream) {
  using namespace okc;
  if (rows < 1 || rows > MMAX || experts < 1 || experts > 32 || grid < 1 || grid >= WGS_SLOTS)
    return int(cudaErrorInvalidValue);
  Params p{};
  p.wAs = w13; p.wBs = w2; p.gsA = scale; p.lut = reinterpret_cast<const uint32_t*>(table);
  p.x = x; p.topk_ids = ids; p.out = out; p.M = rows; p.nloc = experts;
  p.ctr = counters; p.doneA = counters + 8; p.xready = counters + 8 + 512;
  p.ybuf = fix;
  p.doneAs = reinterpret_cast<int*>(fix);
  p.xA = tiles; p.xB = tiles + size_t(WGS_XROWS) * HD;
  constexpr size_t shared = ((sizeof(Smem) + 127) / 128) * 128 +
      (size_t)WGS_NS * UNIT_BYTES + (size_t)WGS_NX * WGS_XS;
  void* args[] = {&p};
  return int(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(moe_fused<false>),
                                         grid, NTL, args, shared, stream));
}

extern "C" size_t expert_dynamic_shared_bytes() {
  using namespace okc;
  constexpr size_t bytes = ((sizeof(Smem) + 127) / 128) * 128 + size_t(WGS_NS)*UNIT_BYTES + size_t(WGS_NX)*WGS_XS;
  static_assert(bytes <= 227 * 1024, "shared memory exceeds Hopper block capacity");
  return bytes;
}
