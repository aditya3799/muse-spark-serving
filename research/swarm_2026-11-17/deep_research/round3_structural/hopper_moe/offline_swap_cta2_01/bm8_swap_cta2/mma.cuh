  static constexpr MmaType kMmaType = MmaType::MMA;
  using MmaShape = Shape<16, 8, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using DRegisters = float[4];

  CUDA_INLINE
  static void fma(uint32_t *a, uint32_t *b, float *c, float *d) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
         "r"(b[0]),  "r"(b[1]),
         "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
  };