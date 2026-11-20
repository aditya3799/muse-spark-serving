# Read physical full-attention pages through the native FA3 batch map

The ordinary page-size-one request pool already stores the physical full-cache slot at each token position. The experimental variant binds that global int32 table to FA3 and supplies a graph-owned int32 cache_batch_idx vector. Its metadata kernel writes this vector and all logical cache lengths. Full attention reads the same columns from the same request row via the native FA3 index. The sliding layer keeps the packed, per-batch SWA table and refreshes its suffix. Each request launches at most ceil((window_left + query_width)/256) + 1 metadata CTAs. The original capacity remains an explicit column slice, preserving the captured full-attention scheduler geometry.

The pinned FA3 host validates cache_seqlens against logical query batch and permits a different page-table row count when kv_batch_idx is provided. Its Hopper mainloop looks up the physical batch row once and the paged manager uses that row to construct its page-table view. The model calls neither append-KV nor per-KV-head descales on this path. [FA3 host checks](https://github.com/sgl-project/sgl-flash-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp) and [Hopper operand binding](https://github.com/sgl-project/sgl-flash-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp).

The live table changes its ownership boundary. The scored backend publishes shared-read completion after the metadata stage in a captured graph. Later scheduler writes could then overlap full-attention layers reading the table. This variant publishes after graph replay on eligible target decode and verify. DFlash inherits the target as its last shared-read runner. The CUDA scheduler consumes that runner's event before writing shared request inputs. The graph holds the same row-ID tensor across successive forwards. An unbounded reuse refreshes the entire SWA table. Real H200 host/scheduler overlap costs and the additional FA3 row-map read require service measurement.

Cumulative patch d947a75635a282efe143dcf9cbd7c33e1d66be55c35eea8d051347093800f8e1 applies against the scored parent. CPU traces exercise 48 real decode/extend argument branches, repeats and duplicate IDs, full-table width preservation, mixed query bounds and all six target mode/barrier combinations. All nine native/SWA/direct x q1/q8/q13 configurations compile to SM90a cubins with the scored int64 map type. Hardware execution of the FP8 FA3 core requires Hopper. No launch on the challenge service was requested.

real_layout.py times all changed metadata builders on the same local device with the scored context capacity, full and SWA slot capacities and dtype layout. It uses disjoint maximum request allocations over three placement regimes. The actual service's per-request page tables were unavailable. It keeps both whole-table length/address checks and deliberately poisons unneeded materialized full and SWA rows. The final two-stream witness exercises the ownership dependency under changed request tables.

## Local scored-layout replay

Pilot and independent holdout passed all 15 shapes each. Across ten changed graph replays per shape, the selected SWA reads matched the parent 5,203,860 times on seed 653779 and 5,202,317 times on seed 953257. Full-request row IDs, all bounded lengths, full-table address equivalence, unreachable poisoning and the independent two-stream ownership witness passed. Peak PyTorch allocations were 239.09 and 239.62 MiB, with 290 and 296 MiB reserved under a 512 MiB cap. The full-to-SWA map uses a distinct positive slot for each reachable suffix token, zero inside evicted prefixes and the allocator's trailing −1 sentinel. The actual hidden service's allocation order is unknown. [Pilot](real_pilot.json) · [Holdout](real_holdout.json)

| RTX metadata at C32, width 8, 74,160 maximum context, independent seed | Parent µs | Suffix µs | Direct µs | Parent-copy µs |
|---|---:|---:|---:|---:|
| Consecutive slots | 4,989.61 | 2,175.40 | 56.79 | 4,997.21 |
| Shuffled 512-token extents | 5,040.34 | 2,194.39 | 55.04 | 5,051.14 |
| Individually shuffled slots | 10,557.65 | 2,463.96 | 498.52 | 10,556.89 |

The 15-shape, equal-weight local comparator kept suffix-only on each seed, at 2.6809× [2.6297, 2.7403] and 2.7158× [2.6610, 2.7707]. It kept direct row lookup at 19.9790× [19.4440, 20.5436] and 21.8564× [21.1770, 22.5223]. These four-arm measurements cover one metadata launch per target forward. Full FA3 now performs an extra row lookup, and safe scheduler writes wait for the target replay to end. Their costs and whole-system H200 throughput are outside this timing. [Suffix-only reward](suffix_reward.json) · [Direct-row reward](direct_reward.json)

