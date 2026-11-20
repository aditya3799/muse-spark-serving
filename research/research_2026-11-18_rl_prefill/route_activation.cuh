#include "elementwise/activation.cuh"

namespace sglang {
struct RouteActivationParams {
  const void* input;
  void* output;
  const void* scales;
  const int32_t* ids;
  uint32_t rows;
  uint32_t experts;
};

// Match the scored Triton BF16 scaling boundary even with fast-math activation.
SGL_DEVICE float route_scale_product(float value, float scale) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "f"(scale));
  return result;
}

template <typename T, bool kUsePDL, uint32_t kGroup>
__global__ void route_scaled_activation(const __grid_constant__ RouteActivationParams p) {
  using namespace device;
  constexpr uint32_t kValues = kMaxVecBytes / sizeof(T);
  constexpr uint32_t kVectors = 4096 / kValues;
  constexpr uint32_t kThreads = 256;
  using vec_t = AlignedVector<T, kValues>;
  const uint32_t base_row = blockIdx.x * kGroup;
  const uint32_t lane = threadIdx.x % 32;
  PDLWaitPrimary<kUsePDL>();
  const int expert = lane < kGroup && base_row + lane < p.rows ? p.ids[base_row + lane] : -1;
  uint32_t active = __ballot_sync(0xffffffffu, expert >= 0 && uint32_t(expert) < p.experts);
  // Each warp visits the same live rows. No CTA synchronization or CPU count read.
  while (active) {
    const int slot = __ffs(active) - 1;
    const int selected = __shfl_sync(0xffffffffu, expert, slot);
    const float scale = device::cast<fp32_t>(static_cast<const T*>(p.scales)[selected]);
    const uint32_t row = base_row + slot;
#pragma unroll
    for (uint32_t vector = threadIdx.x; vector < kVectors; vector += kThreads) {
      const uint32_t offset = row * (2 * kVectors) + vector;
      const vec_t gate = device::load_as<vec_t>(p.input, offset);
      const vec_t up = device::load_as<vec_t>(p.input, offset + kVectors);
      vec_t result;
#pragma unroll
      for (uint32_t i=0; i<kValues; ++i) {
        const T rounded_gate = device::cast<T>(route_scale_product(device::cast<fp32_t>(gate[i]), scale));
        const T rounded_up = device::cast<T>(route_scale_product(device::cast<fp32_t>(up[i]), scale));
        const float activation = apply_activation_f32<ActivationKind::kSiLU>(device::cast<fp32_t>(rounded_gate));
        result[i] = device::cast<T>(activation * device::cast<fp32_t>(rounded_up));
      }
      device::store_as<vec_t>(p.output, result, row * kVectors + vector);
    }
    active &= active - 1;
  }
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL, uint32_t kGroup>
struct RouteActivationKernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
                  tvm::ffi::TensorView ids, tvm::ffi::TensorView scales) {
    using namespace host;
    static_assert(kGroup > 0 && kGroup <= 32);
    auto rows = SymbolicSize{"rows"};
    auto experts = SymbolicSize{"experts"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({rows, 8192}).with_dtype<T>().with_device(device_).verify(input);
    TensorMatcher({rows, 4096}).with_dtype<T>().with_device(device_).verify(output);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device_).verify(ids);
    TensorMatcher({experts}).with_dtype<T>().with_device(device_).verify(scales);
    constexpr uint32_t kVectors = 4096 / (device::kMaxVecBytes / sizeof(T));
    RuntimeCheck(rows.unwrap() <= std::numeric_limits<uint32_t>::max() / (2 * kVectors), "row offsets overflow");
    RuntimeCheck(experts.unwrap() > 0, "invalid expert count");
    if (rows.unwrap() == 0) return;
    const RouteActivationParams params{input.data_ptr(), output.data_ptr(), scales.data_ptr(),
      static_cast<const int32_t*>(ids.data_ptr()), static_cast<uint32_t>(rows.unwrap()),
      static_cast<uint32_t>(experts.unwrap())};
    LaunchKernel(div_ceil(params.rows, kGroup), 256u, device_.unwrap())
      .enable_pdl(kUsePDL)(route_scaled_activation<T, kUsePDL, kGroup>, params);
  }
};
}  // namespace sglang
