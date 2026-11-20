# Decode page-table timing with attention

The scored Humming patch remains the common parent. On one RTX 3050, reading full-attention rows through the request table and translating only the visible sliding suffix improved the combined metadata and two-layer attention measurement by **1.0610×** on discovery and **1.0601×** on independent inputs. Equal weight was assigned to each of the four workloads. The respective paired 95% intervals were 1.0570–1.0652 and 1.0585–1.0617. Across eight replays per case, each seed compared 6,946,816 normalized full/sliding output values, all equal bitwise. The duplicate-parent arm ran the original kernel. [Signed comparison](metadata_attention_reward.json) · [Pilot](metadata_attention_678301.json) · [Independent trial](metadata_attention_1419911.json)

| Concurrent requests and maximum context | Parent µs on independent seed | Candidate µs | Duplicate parent µs |
| --- | ---: | ---: | ---: |
| 1, 2,048 | 2,725 | 2,600 | 2,639 |
| 4, 8,192 | 4,104 | 3,922 | 4,172 |
| 16, 74,160 | 49,974 | 46,183 | 49,896 |
| 32, 74,160 | 98,350 | 90,848 | 98,270 |

These are medians of the CUDA event times. Each graph executes the actual page-table builder once, one full GQA attention and one sliding GQA attention on eight query rows per request. The FP8 cache has two KV heads per rank, 64 features and a physically shuffled 512-token extent layout. The full and sliding allocation capacities come from the official parent. The Ampere implementation uses a lookup for FP8 values and FP16 tensor-core operands. The actual Hopper FA3 timing, TP8 traffic, other 60 layers and scheduler event delay lie outside this comparison. The 20–22× earlier result measures the metadata operation alone. [Allocator-only results](../../research_2026-11-18_rl_prefill/decode_pages/global_rows/FINDINGS.md) · [Portable attention implementation](tree_attention.py)

The model has 16 full and 46 sliding layers. A dedicated `--layer-stack` run visits all of them in order with independent query inputs, while sharing a bounded pair of K/V banks to fit the local card. It retains the once-per-target-forward cost of building the metadata. [Meta model settings](https://dev.meta.ai/docs/muse-spark-open/settings) · [Benchmark driver](metadata_with_attention.py)
