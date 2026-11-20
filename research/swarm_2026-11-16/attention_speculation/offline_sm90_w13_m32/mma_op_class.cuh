  static constexpr MmaType kMmaType = MmaType::WGMMA;
  using MmaShape = Shape<24, 64, 16>;

  using ValTypeC = float;
  using ValTypeD = float;

  static constexpr uint32_t kATypeBits = 16;
  static constexpr uint32_t kBTypeBits = 16;
  static constexpr uint32_t kCTypeBits = 32;
  static constexpr uint32_t kDTypeBits = 32;
  static constexpr bool kNativeMixed = false;

  using BRegisters = uint32_t[4];
  using CRegisters = float[12];
  using DRegisters = float[12];

  CUDA_INLINE
  static void fma(uint64_t &desc, uint32_t *b, float *d, bool pred = true) {
    asm volatile(
      "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %17, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      :  "r"(b[0]),  "r"(b[1]),  "r"(b[2]),  "r"(b[3]),
         "l"(desc), "r"((uint32_t)pred)
    );
  };