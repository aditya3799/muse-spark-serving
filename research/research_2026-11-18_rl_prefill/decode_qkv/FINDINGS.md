# FA3 target decode and verification entry

The candidate dispatches the scored Muse QK-normalization and RoPE kernel as the FA3 query and physical KV writer. It rounds Q and K to BF16 at the parent’s boundary and emits the E4M3FN representation from a 64 KiB table made with the device’s own Torch cast. Value bytes come directly from the BF16 projection. The cache slot-zero behavior and invalid-slot trap follow the native cache writer. Timing ends when FA3 receives its final query and physical cache buffers.

| Local RTX 3050 Laptop GPU, holdout | Scored entry µs | Fused entry µs | Parent-copy µs |
| --- | ---: | ---: | ---: |
| Decode, 1 request, rotary | 40.93 | 17.21 | 41.46 |
| Decode, 16 requests, rotary | 49.64 | 18.19 | 49.58 |
| Verify, 1 request × 8, rotary | 48.46 | 17.46 | 47.93 |
| Verify, 16 requests × 8, no rotary | 76.52 | 44.25 | 76.77 |
| Verify, 32 requests × 8, rotary | 209.12 | 91.83 | 218.34 |

Ten input shapes received equal weights. The kernel-autoresearch comparator gave 2.3490× [2.2500, 2.4489] on seed 167927 and 2.3106× [2.2203, 2.3993] on seed 541859. Signed rewards were +85.40 and +83.75. Each shape had ten balanced rounds for original, candidate and duplicate-original CUDA graphs. Each sample replayed a graph 36 times. The 20 shape/seed pairs matched every byte of Q, original BF16 K, and the entire physical K and V stores on random and changed graph inputs. Graph replay changed addresses, positions and values, including sink slots, overflow, infinities, signed zero and NaNs. Sixteen downstream FP32 paged reference comparisons were bitwise equal for 2,048/74,160-token contexts, full/sliding and query widths 1/8.

## Scored activation and scope

The 317c88a5fd474cd2 startup log records default FA3, FP8 E4M3 KV, page size 1, no unified memory and eight target query rows per request. The model initializes its unquantized RadixAttention without a KV quantization configuration, so the scale fields start as None. The builder emits cumulative patch b6ad473ec72b84648735fc4d4a90fd7a3f3aa6e2ba54004aee5c92c12c856644 against the pinned Git revision. Full package apply, 60 traced parent/candidate forwards, 14 explicit fallback cases and the full and SWA slots all passed. The 64 KiB conversion table is shared across the layers of each eligible target attention backend. Runtime per-layer activation counters require the target hardware.

The pinned Hopper compiler produced all three tested SM90a cubins, including PDL and bounds traps. The pinned FA3 core accepts only FP16 or BF16 on Ampere/Ada. The local result covers the real entry operations with identical downstream attention inputs. End-to-end target throughput, total FA3 latency and H200 stage time remain unmeasured. The raw [donor attention benchmark](https://github.com/gauravjain14/muse-okc-submissions/tree/main/muse_attention_decode/01_fused_flat_splitkv_fp8) used a different track. Its 375.5 µs H200 full verify main kernel at 32 requests illustrates why the local stage gain cannot be applied to the whole service.

## Reproduction

Run build.py, preflight.py and dispatch.py from this directory. Use the CUDA 13 pinned local Python environment and the launcher settings used by the existing Humming research. Run bench.py --seed 167927 --out pilot.json and repeat with seed 541859 and holdout.json. Run reward.py for the plugin comparison and offline.py for Hopper code generation. The parent trace executes the scored Triton QK/RoPE body, the stock BF16-to-FP8 conversions, the pinned native 128-byte store_cache operation and prepare_paged_mha_query. The candidate trace executes the compiled packet writer and the same prepare method. No external run was requested.
