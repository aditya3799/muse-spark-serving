// Fuse global BF16 scaling with the pinned stock CUDA SiLU implementation.
#include "elementwise/activation.cuh"

namespace sglang {
struct ScaledActivationParams {
  const void* input;
  void* output;
  const void* scales;
  const int32_t* expert_ids;
  uint32_t hidden;
  uint32_t rows;
  uint32_t experts;
};

// Stock Triton scaling does not flush FP32 subnormals. Explicit PTX prevents
// --use_fast_math (needed to match stock activation) from altering this multiply.
SGL_DEVICE float exact_scale_product(float value, float scale) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(value), "f"(scale));
  return result;
}

template <typename T, bool kUsePDL>
__global__ void scaled_silu_kernel(const __grid_constant__ ScaledActivationParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;
  const uint32_t vectors = params.hidden / kVecSize;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t row = tid / vectors;
  if (row >= params.rows) return;
  const int32_t expert = params.expert_ids[row];
  if (expert < 0 || static_cast<uint32_t>(expert) >= params.experts) return;
  PDLWaitPrimary<kUsePDL>();
  const auto offset = row * vectors * 2 + tid % vectors;
  const auto gate = device::load_as<vec_t>(params.input, offset);
  const auto up = device::load_as<vec_t>(params.input, offset + vectors);
  const float scale = device::cast<fp32_t>(static_cast<const T*>(params.scales)[expert]);
  vec_t result;
#pragma unroll
  for (int i = 0; i < kVecSize; ++i) {
    const T scaled_gate = device::cast<T>(exact_scale_product(device::cast<fp32_t>(gate[i]), scale));
    const T scaled_up = device::cast<T>(exact_scale_product(device::cast<fp32_t>(up[i]), scale));
    const float gate_f32 = device::cast<fp32_t>(scaled_gate);
    const float up_f32 = device::cast<fp32_t>(scaled_up);
    result[i] = device::cast<T>(apply_activation_f32<ActivationKind::kSiLU>(gate_f32) * up_f32);
  }
  device::store_as<vec_t>(params.output, result, tid);
  PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL>
struct ScaledActivationKernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output,
                  tvm::ffi::TensorView ids, tvm::ffi::TensorView scales) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"input_width"};
    auto hidden = SymbolicSize{"hidden"};
    auto experts = SymbolicSize{"experts"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({rows, width}).with_dtype<T>().with_device(device_).verify(input);
    TensorMatcher({rows, hidden}).with_dtype<T>().with_device(device_).verify(output);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device(device_).verify(ids);
    TensorMatcher({experts}).with_dtype<T>().with_device(device_).verify(scales);
    constexpr uint32_t vector_size = device::kMaxVecBytes / sizeof(T);
    RuntimeCheck(hidden.unwrap() > 0 && hidden.unwrap() * 2 == width.unwrap(), "invalid widths");
    RuntimeCheck(hidden.unwrap() % vector_size == 0, "hidden width must align to vector size");
    RuntimeCheck(experts.unwrap() > 0, "empty scales");
    const uint64_t items = rows.unwrap() * (hidden.unwrap() / vector_size);
    RuntimeCheck(items <= std::numeric_limits<uint32_t>::max() / 2, "32-bit indexing overflow");
    if (rows.unwrap() == 0) return;
    const ScaledActivationParams params{input.data_ptr(), output.data_ptr(), scales.data_ptr(),
      static_cast<const int32_t*>(ids.data_ptr()), static_cast<uint32_t>(hidden.unwrap()),
      static_cast<uint32_t>(rows.unwrap()), static_cast<uint32_t>(experts.unwrap())};
    LaunchKernel(div_ceil(static_cast<uint32_t>(items), 256u), 256u, device_.unwrap())
      .enable_pdl(kUsePDL)(scaled_silu_kernel<T, kUsePDL>, params);
  }
};
}  // namespace sglang