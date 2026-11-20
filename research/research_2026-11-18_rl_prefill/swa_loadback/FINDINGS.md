# Sliding-window host restore

The pinned Rust and Python cache implementations restore all SWA positions in an intersecting host node whenever the requested window reaches any part of that node. The experiment uses the existing node-split operation to establish the page-aligned start of the actual attention window, then repeats the authoritative match. The original full-cache branch, earlier host contents and allocation ownership survive the split. Existing split actions update write-through acknowledgements before the ordinary transfer is built. The source-only delta applies to the scored baseline. [Delta](delta.diff) [Pinned Rust transfer builder](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/rust/sglang-radix-tree/src/components/swa.rs) [Pinned Rust edge split](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/rust/sglang-radix-tree/src/unified_tree_core.rs)

| Fixture | Parent host positions | Candidate host positions |
| --- | ---: | ---: |
| One 16,384-position host node, 2,048 window | 16,384 | 2,048 |
| Same host node followed by 220 GPU-resident positions | 16,384 | 1,828 |
| Already-split 2,048 terminal node | 2,048 | 2,048 |
| Fully GPU-resident | 0 | 0 |
| 2,047 window, 64-position pages, 256 GPU-resident positions | 16,384 | 1,792 |

The transfer builder produced the same visible slot IDs, split Full and SWA host/device indices without reordering and preserved an in-flight write receipt. A second identical match did not split again. The local test used pinned Python tree and component methods, torch tensors, a single-branch walker and synthetic LRU observers. It passed 100 randomized fixtures in addition to the five cases above. The native Rust binding and host-device DMA were not exercised locally. [Recorded comparisons](transfer_results.json) [Exact source methods and fixture](test_transfer.py)

The judge's host layout has 11,776 SWA bytes per position per rank. The first fixture saves 161 MiB of host-to-device traffic per rank. Fresh terminal leaves are already split at a whole attention window in the Rust core, which explains the unchanged third case. Earlier interior branches and incompletely resident paths can expose an oversized node. The 32-user H200 receipt reports aggregate cache hit 0.92 and supplies neither per-tier hits nor the sizes of host nodes actually restored. The earlier capacity experiment considered a smaller SWA allocation and measured the resulting token capacity algebraically. Neither result supports an H200 end-to-end gain from this fix. [H200 result](../../release_2026-11-18/scored_submission_status.json) [Previous capacity calculation](../../research_2026-11-18_prefill_route_windows/cache_capacity.json)

