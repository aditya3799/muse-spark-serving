# K16 expanded weights and K64 packed scales

A circuit-preserving representation exists for each tensor-core operand. Every E2M1 nibble has an exact E4M3 encoding, including signed zero. Using the expanded nibble as the unscaled base, multiplying by an independently decoded BF16 group scale divided by four matched all 4,064 finite nibble/scale inputs against the frozen Humming operand in the CPU reference. [Arithmetic and scratch](scratch_costs.json)

Expanding weights to eight bits resolves the 128-bit TMA-layout alignment at CTA K16, but NVIDIA's grouped mixed-input consumer still requires at least four GMMA K-blocks per pipeline tile. BF16 WGMMA advances by 16 K elements, leaving only one at K16. The frozen upstream compile failed specifically on the four-block assertion. The consumer also explicitly preloads the next K fragment and releases barriers based on the number outstanding. Removing the assertion alone cannot implement the K16 path. [Pinned source](https://github.com/NVIDIA/cutlass/blob/cd9d8a7d612b6b9ad0f1b698925e31243e828629/include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp#L943-L1005) [Actual NVCC diagnostics](fp8_compile.json)

## Offline SM90a controls

CUDA 13.0 compiled a complete instantiation of the Hopper grouped device kernel, with transposed problem axes, 128×16×K CTAs, cooperative pointer arrays, three TMA stages, FP32 accumulators and BF16 output. The custom stage loads one uint32 for every four E4M3 scales grouped across 64 K elements per output channel. Each 16-K GMMA fragment selects its own byte, converts it to BF16, multiplies by 0.25 and scales the exact FP4 or expanded FP8 operand in BF16. This prototype specializes the private compiled scale tag in its own include overlay. It does not install a modified CUTLASS. [Compiled source](fp8_probe.cu) [Custom fragment](custom_fragment.diff) [Reproduction](compile_custom.py)

| Grouped BF16 kernel | Result | ptxas registers and barriers | Cubin bytes |
| --- | --- | --- | ---: |
| Ordinary FP8 weight, CTA K16 | Rejected on the four-K-block assertion | — | — |
| Ordinary FP8 weight, CTA K64, single BF16 scale per tile | Built | 168, 2 | 93,136 |
| Expanded FP8 weight, CTA K64, four independent E4M3 scales | Built | 168, 2 | 95,184 |
| Packed E2M1 weight, CTA K64, four independent E4M3 scales | Built | 168, 2 | 103,712 |

The K64 scale-word proof covers every finite signed scale byte in every selection lane and compares 65,024 operand values with the parent. The compiled packed-FP4 kernel avoids weight widening. The existing adapter still needs fixed-size expert-major route descriptors, GPU-populated group pointers and CUTLASS's packed layout. The actual grouped device kernel was not executed because the local GPU is SM86; static compilation does not establish final routed W13, W2, or whole-FFN parity or speed. Matching Humming's order of FP32 partial sums, BF16 rounding, global-scale restoration and finalized routing on SM90a remains a hardware test. [Finite-domain/selector results](screen.json) [Exact compiler revision and cubin hashes](fp8_compile.json)

## Bounded scratch and traffic

A scratch slab reused between W13 and W2 decouples capacity from the number of resident layers. For all 32 local experts at H=I=4,096, the ordinary expanded design needs 1,152 MiB W13 or 576 MiB W2 (FP8 weights plus independently expanded BF16 scales). If every routed layer is converted once, reading the original and writing scratch takes 2,592 MiB per layer before GEMM reads. At an ideal 4.8 TB/s, and only when those bytes cross HBM, 60 layers would take at least 34.0 ms of transfer service; overlap could hide some on a serving timeline. This does not include grouped route row buffers or reductions. [Stage-specific bytes](scratch_costs.json)

The packed-FP4 custom branch has a distinct budget. If the CUTLASS row-major FP4 weights and N-major uint32 scale words have to be rebuilt from Humming's per-lane layout each time, a W13 slab uses 512 MiB weights plus 64 MiB scales, and W2 uses 256 MiB plus 32 MiB. Their sequential peak is 576 MiB. Reading and repacking both full banks moves 1,728 MiB per layer before the grouped GEMM reads them, equivalent to 22.65 ms for 60 layers at the same ideal bandwidth assumption. Retaining both complete arrangements across all 60 layers would still be unworkable. A shared native layout or an appropriately measured streaming repacker is needed for a serving candidate.
