# Sliding-window metadata in target decode and verification

The scored FA3 replay constructs full and SWA physical page tables from every live context row. The candidate keeps every full-attention page, the logical lengths and prefix sums. It only refreshes SWA pages from the earliest window any query can read. It preserves the entire first intersecting 256-page block. The two attention branches continue using the same capacity, absolute sequence lengths, split policy and physical slot at every FA3-readable position.

The pinned FA3 block planner offsets K by max(0, m_idx + seqlen_k − seqlen_q − window_left) on local attention. The Hopper paged loader receives leftpad_k + n_offset. In the model's direct path the left pad is zero. The scheduler and full-attention kernel retain their original input lengths and page capacity. Unqualified FA variants, unified/translated pools, sharded KV, local attention builders, ragged verification, prefill and non-Hopper devices use the parent path. [Block planner](https://github.com/sgl-project/sgl-flash-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/block.h) and [Hopper paged mainloop](https://github.com/sgl-project/sgl-flash-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp).

| RTX 3050 Laptop GPU holdout | Parent µs | Candidate µs | Parent copy µs |
| --- | ---: | ---: | ---: |
| 1 request, ≤2,048 tokens, decode | 21.59 | 18.82 | 22.66 |
| 1 request, ≤74,160 tokens, verify width 8 | 288.64 | 35.50 | 291.29 |
| 4 requests, ≤74,160 tokens, verify width 8 | 1,150.76 | 339.20 | 1,149.27 |
| 16 requests, ≤80,000 tokens, verify width 8 | 5,073.37 | 1,424.81 | 5,070.89 |
| 32 requests, ≤80,000 tokens, verify width 8 | 10,148.74 | 2,841.77 | 10,154.41 |

Each shape changes request IDs, row lengths between one third of the bound and the bound itself, and the full-to-SWA page map on each measured round. Nine shapes, including three requests at query width 3 and seven at width 13, ran on each of two seeds. Full page tables and length metadata matched. Every live SWA page address that a causal window may read matched, including tests with deliberately invalid IDs in unreachable prefixes. The first holdout harness required some old prefix entries to remain poisoned even after an earlier short replay could legally populate them. The repaired coverage-only check replants poison after timing in that case. The candidate kernel, seed and timed schedule stayed identical. The short decode case regressed from 23.72 to 29.01 µs in the pilot and improved in the holdout.

The kernel-autoresearch comparator reported 3.3713× [3.1612, 3.5733] and 3.3787× [3.2326, 3.5432]. Both local decisions are KEEP, with signed rewards +121.53 and +121.75. This is one metadata launch per rank per target forward. GPU component timing excludes the FA3 body, other layers, draft processing and inter-rank traffic. H200 service speedup remains unmeasured.

Run build.py and preflight.py. Use bench.py --seed 684461 --out pilot.json and then seed 732719 with holdout.json under the pinned CUDA 13 environment. Pass the resulting files to research_2026-11-18_rl/score_trials.py with --holdout. Local benchmarking directly replays the native source kernels on SM86. The integrated branch selects only the pinned Hopper pathway whose FA3 physical read bounds were audited. The cumulative candidate SHA-256 is 1fcc5661c9896094f917d4e7d08c10c014d62fc3e5bf1bceb4fbf83bcbe5844c.

The first two trials use a synthetic int32 full-to-SWA map with 524,288 entries and uniformly scattered request slots. The pinned allocator uses an int64 map with 1,893,608 full slots for the scored H200. The direct-row follow-on’s real_layout.py compares the same suffix patch using that storage layout and disjoint contiguous, 512-token extent, and fully fragmented requests. Use those separate receipts to judge sensitivity to address locality.
