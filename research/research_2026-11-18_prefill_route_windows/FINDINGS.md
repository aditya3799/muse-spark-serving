# Global expert routing for large prefills

The proposed path aligns all local routes once, preserving every row's original top-eight slot. It groups up to 64 routes per expert tile. Up to 48 such groups share the existing 3,072-row activation workspace per window. The small-batch path keeps the packaged route builder. The large-batch path uses the same packed W13/W2 bank, activation, global scales, finalizer and original output addresses. Sorting executes on the device. Fixed-capacity empty launches return after reading the device-side padded count. This experiment targets the cost of rebuilding sparse expert tiles independently in each 384-token chunk.

On a 16,384-token uniform routing fixture, the original path issued 1,376 full expert-weight tile sweeps and the new path issued 268. On the contiguous skew fixture from the prefill donor, the counts were 487 and 486. In a random permutation of the same skew histogram the counts were 748 and 486. These are requested weight streams. HBM cache behavior, launch costs and matrix multiply throughput affect elapsed time.

| Rows | Contiguous donor fixture | Shuffled donor histogram | Uniform global top-8 |
| ---: | ---: | ---: | ---: |
| 753 | 0.994x | 1.024x | 1.085x |
| 3,012 | 0.975x | 1.036x | 1.228x |
| 16,384 | 0.969x | 1.046x | 1.250x |

The ratios compare medians of paired GPU event timings for complete routed W13, SwiGLU and W2 on the same data and weight bank. The global path's sort and both paths' bounded window launches are included. Each smaller case used four alternating-order pairs and each 16,384-row case used two. All nine compared outputs passed, with worst relative L2 below 0.000123 over live routes. This local RTX 3050 exercise replaces native Hopper warpgroup MMA and TMA with warp MMA, global loads and three ordered stages in both paths. The input distributions are synthetic. Native sm_90a compiles offline and requires Hopper execution to measure end-to-end. The tested loop omits the shared finalizer on both sides. A production policy still requires held-out timing and actual routing. These results do not justify promoting a global default or claiming the 1.8x serving gate.

build_windows.py renders and compiles the isolated CUDA sources against the exact submitted Hopper package. render_adapter.py emits its experimental SGLang backend with the new native call. The GPU test in run_windows.py uses the pinned SGLang aligner and the full two-projection participant from the numerical qualification as its starting point. Its raw samples are in gpu_abba.json and gpu_cold_prefill.json. route_bounds.py checks slot identity and workspace bounds across 40 route distributions, including all-remote routes. It stores the counts and relative scheduler proxy in route_bounds.json.

## Prefix reuse and KV capacity

The scored parent used an average first prompt of 74,160 tokens, twelve later additions of 753 tokens and 220 output tokens per turn. Full retention gives an analytical cache hit of 92.00018 percent under mean lengths. The official result gives 92 percent at every load. The cold first prompt accounts for 89.14 percent of unique prompt tokens. Match rates include CPU-backed hits, so capacity can still affect latency at high concurrency.

Rank-zero startup logged 1,893,608 full GPU tokens and 1,514,886 sliding-window tokens. The fixed ratio was 0.8. The full, sliding-window and separate draft pools use 4,096, 11,776 and 2,560 bytes per token, respectively, as independently derivable from the three host allocations. The 32 mean final-turn prompts need approximately 2,746,752 full KV positions. A ratio of 0.35 would fit them with roughly 77,916 full positions to spare at the current byte budget and leave 988,633 sliding-window slots. The 16-user mean already fits the original full pool. The capacity calculation in cache_capacity.json excludes scheduling differences, variable request lengths, evictions, CPU transfer timing and launch overhead. Direct full-model measurements would be needed to choose a pool ratio.
