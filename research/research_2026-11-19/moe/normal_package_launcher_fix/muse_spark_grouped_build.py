# SPDX-License-Identifier: Apache-2.0
"""Local build of one pinned CUTLASS SM90a grouped NVFP4 kernel; no downloads."""
import ctypes as ct
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import signal
import subprocess
import tempfile
from functools import lru_cache

LOG = logging.getLogger(__name__)
HEADER_SHA = '6bc2a648b879b8555c4c3337021ac39906befa4aaa0865c47866058f698ae562'
GROUP_SHA = 'ae3a9a82c245c068e30d71dae6f6877bb02080e1914d168a10589107b0f059b3'
REL = 'cutlass/detail/collective/mixed_input_utils.hpp'
GROUP = 'cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp'


def _headers():
    for package, suffix in [('deep_gemm', 'include'), ('flashinfer', 'data/cutlass/include')]:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None or spec.origin is None:
                continue
            path = Path(spec.origin).parent / suffix
            if (hashlib.sha256((path/REL).read_bytes()).hexdigest() == HEADER_SHA and
                    hashlib.sha256((path/GROUP).read_bytes()).hexdigest() == GROUP_SHA):
                return path
        except (ImportError, OSError):
            continue
    raise RuntimeError('exact pinned CUTLASS headers unavailable; keep original Humming')


CORE_LIMIT_WRAPPER = """import os, resource, sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
try:
    with open('/proc/self/coredump_filter', 'w') as stream:
        stream.write('0')
except OSError:
    pass
os.execv(sys.argv[1], sys.argv[1:])
"""


def _compiler_process(args):
    # No Python preexec callback in a forked multithreaded Torch worker.
    # A fresh interpreter applies child-local limits, then execs NVCC in-place.
    return subprocess.Popen([sys.executable, '-c', CORE_LIMIT_WRAPPER, *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


@lru_cache(maxsize=1)
def library_path():
    from torch.utils.cpp_extension import CUDA_HOME
    if not CUDA_HOME:
        raise RuntimeError('CUDA_HOME unavailable')
    home = Path(CUDA_HOME)
    include = _headers()
    raw = (include/REL).read_text()
    if raw.count(NEEDLE) != 2:
        raise RuntimeError('unexpected pinned scale conversion anchors')
    fixed = raw.replace(NEEDLE, CUSTOM, 1)
    digest = hashlib.sha256(('normal-g16-t64-s3-v1'+SOURCE+fixed+str(home)).encode()).hexdigest()[:20]
    cache = Path(tempfile.gettempdir()) / ('sglang_muse_grouped_' + digest)
    cache.mkdir(exist_ok=True)
    target = cache/'kernel.so'
    with (cache/'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        failed = cache/'build.failed'
        if failed.exists():
            raise RuntimeError('cached grouped build failure: '+failed.read_text()[-1500:])
        if not target.exists():
            overlay = cache/'overlay'/REL
            overlay.parent.mkdir(parents=True, exist_ok=True)
            overlay.write_text(fixed)
            source = cache/'kernel.cu'
            source.write_text(SOURCE)
            libdirs = [home/'lib64', home/'lib', home/'lib64/stubs',
                       Path('/usr/lib/wsl/lib'), Path('/usr/lib/x86_64-linux-gnu')]
            cudart = next((x for q in libdirs for x in q.glob('libcudart.so*') if x.is_file()), None)
            driver = next((x for q in libdirs for x in q.glob('libcuda.so*') if x.is_file()), None)
            if cudart is None or driver is None:
                raise RuntimeError('CUDA runtime/driver link libraries unavailable')
            args = [str(home/'bin/nvcc'), '-std=c++17', '-arch=sm_90a', '--shared',
                    '-Xcompiler=-fPIC', '--expt-extended-lambda', '--expt-relaxed-constexpr',
                    '-O2', '-Xptxas=-v', '-DPACKED_FP4', '-DPACKED_SCALE',
                    '-DPROBE_CTA_K=64', '-DPROBE_TOKEN_TILE=64',
                    '-I'+str(cache/'overlay'), '-I'+str(include), '-I'+str(home/'include/cccl'), '-cudart=none',
                    str(source), '-L'+str(cudart.parent), '-l:'+cudart.name,
                    '-L'+str(driver.parent), '-l:'+driver.name,
                    '-Xlinker=-rpath='+str(cudart.parent), '-o', str(cache/'kernel.tmp.so')]
            version = subprocess.check_output([str(home/'bin/nvcc'), '--version'], text=True)
            process = _compiler_process(args)
            try:
                output, _ = process.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                output += '\nGrouped build exceeded 600 seconds; compiler process group stopped.'
            (cache/'build.log').write_text(output)
            LOG.info('Muse grouped compile: headers=%s nvcc=%s returncode=%s log=%s',
                     include, version.strip().splitlines()[-1], process.returncode, cache/'build.log')
            if process.returncode:
                failed.write_text(output[-1500:])
                raise RuntimeError('grouped CUDA build failed: '+output[-1500:])
            os.replace(cache/'kernel.tmp.so', target)
    LOG.info('Muse grouped binary: path=%s sha256=%s', target,
             hashlib.sha256(target.read_bytes()).hexdigest())
    return target


def library():
    return ct.CDLL(str(library_path()))

SOURCE = r"""// Standalone research host adapter for the pinned CUTLASS grouped packed-FP4 kernel.
// The caller owns every operand and stream. Create/destroy outside graph capture.
// CUTLASS grouped Hopper R/S mixed-input compile probe based on NVIDIA example 69.
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/device_kernel.h"
using namespace cute;
using Bf16 = cutlass::bfloat16_t;
#ifdef PACKED_FP4
using Quant = cutlass::float_e2m1_t;
#else
using Quant = cutlass::float_e4m3_t;
#endif
using GroupShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
#ifndef PROBE_TOKEN_TILE
#define PROBE_TOKEN_TILE 16
#endif
using ProbeTile = Shape<_128,Int<PROBE_TOKEN_TILE>,Int<PROBE_CTA_K>>;
using Cluster = Shape<_1,_1,_1>;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Epi = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,ProbeTile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,
    float,float,Bf16,Col*,8,Bf16,Col*,8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
#ifdef PACKED_SCALE
using Scale = uint32_t;
#else
using Scale = Bf16;
#endif
#ifdef SHARED_HUMMING_WEIGHT
#include "humming_atom.hpp"
using WeightLayout = HummingWeightLayout;
#else
using WeightLayout = Row;
#endif
// Judge CUTLASS builder passes StageCountType::bytes as the manual-stage template argument.
// Preserve exactly three stages without editing installed headers.
struct ManualStages3 : cutlass::gemm::collective::StageCount<3> { static constexpr int bytes = 3; };
using Main = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    cute::tuple<Quant,Scale>,WeightLayout*,(128 / cutlass::sizeof_bits<Quant>::value),Bf16,Col*,8,float,ProbeTile,Cluster,
    ManualStages3,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<GroupShape,Main,Epi>;
template __global__ void cutlass::device_kernel<Kernel>(CUTLASS_GRID_CONSTANT Kernel::Params const params);

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
  d->ws[g] = make_humming_layout(n,k);
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

extern "C" size_t okc_owned_bytes(void* handle) { auto* c=static_cast<Context*>(handle); return c ? sizeof(Descriptors)+DeviceGemm::get_workspace_size(arguments(*c)) : 0; }
"""
NEEDLE = r"""      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));"""
CUSTOM = r"""      if constexpr (is_same_v<ElementScale, uint32_t> && is_same_v<DstType, cutlass::bfloat16_t>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            const uint32_t word = scales_vm(j, i);
            const uint8_t byte = uint8_t(word >> (8 * k_block));
            const auto scale = cutlass::float_e4m3_t::bitcast(byte);
            const auto normalized_scale = cutlass::bfloat16_t(float(scale) * 0.25f);
            dst_vm(j, i) *= normalized_scale;
          }
        }
      }
      else if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));"""
