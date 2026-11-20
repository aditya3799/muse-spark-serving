// Standalone research host adapter for the pinned CUTLASS grouped packed-FP4 kernel.
// The caller owns every operand and stream. Create/destroy outside graph capture.
#include "fp8_probe.cu"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include <cuda_runtime.h>
#include <new>
#include <cstdint>

using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Problem = Shape<int,int,int>;
#ifdef SHARED_HUMMING_WEIGHT
using WS = HummingWeightLayout;
#else
using WS = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<Row*>>;
#endif
using XS = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<Col*>>;
using DS = Kernel::InternalStrideD;
using SS = Main::StrideScale;
constexpr int kMaxGroups = 32;

struct alignas(128) Descriptors {
  Problem problems[kMaxGroups];
  Quant const* w[kMaxGroups];
  Bf16 const* x[kMaxGroups];
  Bf16 const* c[kMaxGroups];
  Bf16* y[kMaxGroups];
  uint32_t const* scales[kMaxGroups];
  WS ws[kMaxGroups];
  XS xs[kMaxGroups];
  SS ss[kMaxGroups];
  DS ds[kMaxGroups];
};

struct Context {
  DeviceGemm gemm;
  Descriptors* desc = nullptr;
  void* workspace = nullptr;
  cutlass::KernelHardwareInfo hw{};
  int groups = 0, n = 0, k = 0;
};

__global__ void descriptors(Descriptors* d, int groups, int n, int k,
                            uint8_t const* w, uint32_t const* s, Bf16 const* x, Bf16* y,
                            int const* starts, int const* padded) {
  int g = threadIdx.x;
  if (g >= groups) return;
  int64_t base = starts[g];
  d->problems[g] = make_shape(n, padded[g], k);
  d->w[g] = reinterpret_cast<Quant const*>(w + int64_t(g)*n*(k/2));
  d->scales[g] = s + int64_t(g)*(k/64)*n;
  d->x[g] = x + base*k;
  d->y[g] = y + base*n;
  d->c[g] = y + base*n;
#ifdef SHARED_HUMMING_WEIGHT
  d->ws[g] = cute::tile_to_shape(HummingWarpAtom{},make_shape(n,k,_1{}));
#else
  d->ws[g] = WS{k,_1{},_0{}};
#endif
  d->xs[g] = XS{k,_1{},_0{}};
  d->ss[g] = SS{_1{},n,_0{}};
  d->ds[g] = DS{_1{},n,_0{}};
}

static DeviceGemm::Arguments arguments(Context const& ctx) {
  auto* d = ctx.desc;
  DeviceGemm::Arguments args{};
  decltype(args.epilogue.thread) op{};
  op.alpha = 1.0f;
  op.beta = 0.0f;
  op.alpha_ptr = nullptr; op.beta_ptr = nullptr;
  op.alpha_ptr_array = nullptr; op.beta_ptr_array = nullptr;
  op.dAlpha = {_0{},_0{},0};
  op.dBeta = {_0{},_0{},0};
  return {cutlass::gemm::GemmUniversalMode::kGrouped,
          {ctx.groups,d->problems,nullptr},
          {d->w,d->ws,d->x,d->xs,d->scales,d->ss,64},
          {op,d->c,d->ds,d->y,d->ds},ctx.hw};
}

// 1..999 CUDA error numbers; >=1000 indicates a CUTLASS failure.
extern "C" int okc_destroy(void* handle) {
  auto* ctx=static_cast<Context*>(handle);
  if (!ctx) return 0;
  cudaError_t a=ctx->workspace?cudaFree(ctx->workspace):cudaSuccess;
  cudaError_t b=ctx->desc?cudaFree(ctx->desc):cudaSuccess;
  delete ctx;
  return int(a!=cudaSuccess?a:b);
}

extern "C" int okc_create(void** out,int groups,int n,int k,cudaStream_t stream) {
  if (!out || groups<1 || groups>kMaxGroups || n<64 || k<64 || n%64 || k%64)
    return int(cudaErrorInvalidValue);
  *out=nullptr;
  int device,major,minor,sms;
  cudaError_t err=cudaGetDevice(&device);
  if (err==cudaSuccess) err=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
  if (err==cudaSuccess) err=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);
  if (err==cudaSuccess) err=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);
  if (err!=cudaSuccess) return int(err);
  if (major!=9 || minor!=0) return int(cudaErrorInvalidDeviceFunction);
  auto* ctx=new(std::nothrow) Context;
  if (!ctx) return int(cudaErrorMemoryAllocation);
  ctx->hw.device_id=device;
  ctx->hw.sm_count=sms;
  ctx->groups=groups; ctx->n=n; ctx->k=k;
  err=cudaMalloc(&ctx->desc,sizeof(Descriptors));
  if (err!=cudaSuccess) { okc_destroy(ctx); return int(err); }
  auto args=arguments(*ctx);
  auto rc=ctx->gemm.can_implement(args);
  if (rc!=cutlass::Status::kSuccess) { okc_destroy(ctx); return 1000+int(rc); }
  auto bytes=DeviceGemm::get_workspace_size(args);
  if (bytes) err=cudaMalloc(&ctx->workspace,bytes);
  if (err!=cudaSuccess) { okc_destroy(ctx); return int(err); }
  rc=ctx->gemm.initialize(args,ctx->workspace,stream);
  if (rc!=cutlass::Status::kSuccess) { okc_destroy(ctx); return 1000+int(rc); }
  *out=ctx;
  return 0;
}

extern "C" int okc_run(void* handle,uint8_t const* w,uint32_t const* s,Bf16 const* x,Bf16* y,
                       int const* starts,int const* padded,cudaStream_t stream) {
  auto* c=static_cast<Context*>(handle);
  if (!c || !w || !s || !x || !y || !starts || !padded)
    return int(cudaErrorInvalidValue);
  descriptors<<<1,kMaxGroups,0,stream>>>(c->desc,c->groups,c->n,c->k,w,s,x,y,starts,padded);
  auto err=cudaGetLastError();
  if (err!=cudaSuccess) return int(err);
  auto rc=c->gemm.run(stream);
  return rc==cutlass::Status::kSuccess?0:1000+int(rc);
}
