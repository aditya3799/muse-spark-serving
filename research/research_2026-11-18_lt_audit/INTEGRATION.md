# Source patch and public-checkpoint trials

The cumulative research patch updates only allowed SGLang source files. The original DFlashWorkerV2.prewarm_sampling hook runs inside the Scheduler's forward stream after graph capture and weight loading. The callback selects eligible row counts from the configured decode graph batch sizes and draft block width. It builds the native helper locally, asks the device's full-operation cuBLASLt heuristic for reduction settings, checks possible tilings on the current device and current weights, and compares fresh complete outputs and paired timings before allowing a plan. The TP group retains only signatures all participating ranks separately support. Unknown shapes, streams, math flags, unsupported builds and failed comparisons use the parent projection. The hot projection preserves the ordinary full-width bias-free FC and hidden normalization. Updating checkpoint weights discards the plan. There are no embedded local algorithm IDs.

The original fixed-plan research trial on public checkpoint 46c90f412549c472941a68382d91a99e5e6372a0 used its full native BF16 FC, hidden normalization, actual TP-rank-zero K/V output slices from all five layers, each real K normalization, and the original FP32 RoPE cache construction with theta 500000. It compared all FC, normalized, projected K/V, and physical cache outputs under two new seeds and both partial/full commitment. All compared values matched and every upstream perturbation changed computed FC values. The original held-out plugin decision was MIXED despite a 1.0642× aggregate [1.0597,1.0689]. At B1 partial it regressed 2.25% and at B4 full it regressed 0.66%. B16 partial/full ran in 6.93/6.90 ms compared with 7.25/7.26 ms native. B32 partial/full ran in 12.15/11.92 ms compared with 14.82/15.07 ms native. Synthetic changing target residual taps were used throughout. The study did not run the target transformer, all eight distributed ranks, real multi-turn serving or token comparison.

The next candidate froze a shape-general rule to use native FC for fewer than 128 input rows before any projection lookup. Its warmup skips those geometries. This threshold was chosen from the prior local trial. New fully independent activation seeds were fixed at 890761 and 1176971 before testing the updated policy. The dynamic plan selection uses only startup calibration tensors. Timed held-out activation tensors never participate in choosing the plan. The original fixed-plan receipt remains available at [real_reward.json](real_reward.json).

[Integrated source patch](integrated_candidate.diff) · [Cumulative build and pinned-index check](integrate.py) · [Guarded production-method trial](guarded_entry.py) · [Public checkpoint test](real_coupled.py) · [Original fixed-plan evidence](real_holdout_detail.json)


## Frozen guarded production result

The 110,991-byte cumulative patch passed a clean staged application against SGLang 297e54fd105e974cdf401e5b4d20253a22c16f90. Its SHA-256 is `84eff9bc0fd443eba15e3b2f23ca8985e7776c562f6d5746255613df5924ce18`. The scored parent SHA-256 is `8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`. Every changed path is permitted source under python/sglang. The longest added line has 226 characters. The patch contains no compiled binary, rename or model-loader change.

On the real checkpoint the scheduler-stream startup accumulated 13.487 seconds, including local first-use compilation and discovery. It independently admitted 128 and 256 rows. The 8- and 32-row projections took the native branch. All sixteen changing-input test cells matched at the FC, hidden normalization, dense KV and physical KV levels. Each seed checked 347,494,400 cached values per arm. This exercised the exact patched worker prewarm and model methods with fixture implementations for unrelated services and upstream target layers. It ran on one physical TP-rank-zero slice and did not launch the full SGLang serving stack.

The plugin's aggregate pilot verdict was **KEEP**, 1.05868× [1.04942, 1.06729]. The independent held-out verdict was **MIXED**, 1.06609× [1.06059, 1.07172]. Local promotion was false. [Formal paired reward](guarded_reward.json)

### Native-bypass controls

Ratios are paired plugin speedups. Brackets give 95% bootstrap intervals. Each arm was independently executed with the same changing inputs and reverse-order pair schedules.

| Seed | Batch and commit | Candidate / native | Native control / native | Candidate / native control |
|---|---|---:|---:|---:|
| pilot | B1_partial | 0.9848 [0.9433, 1.0122] | 0.9873 [0.9614, 1.0054] | 0.9974 [0.9807, 1.0088] |
| pilot | B1_full | 0.9732 [0.9221, 1.0176] | 1.0057 [0.9957, 1.0167] | 0.9677 [0.9101, 1.0162] |
| pilot | B4_partial | 1.0022 [0.9950, 1.0101] | 1.0146 [1.0076, 1.0217] | 0.9878 [0.9778, 0.9966] |
| pilot | B4_full | 1.0149 [1.0063, 1.0236] | 1.0121 [0.9961, 1.0278] | 1.0027 [0.9890, 1.0165] |
| holdout | B1_partial | 0.9938 [0.9816, 1.0044] | 1.0034 [0.9947, 1.0109] | 0.9904 [0.9815, 0.9978] |
| holdout | B1_full | 1.0036 [0.9897, 1.0241] | 1.0092 [0.9963, 1.0290] | 0.9944 [0.9887, 0.9990] |
| holdout | B4_partial | 0.9871 [0.9813, 0.9912] | 0.9911 [0.9797, 0.9989] | 0.9960 [0.9872, 1.0082] |
| holdout | B4_full | 1.0032 [0.9919, 1.0182] | 0.9952 [0.9882, 1.0019] | 1.0081 [0.9958, 1.0288] |

Held-out B4 partial triggered MIXED. A second execution of the unchanged native control also slowed relative to baseline, while the candidate-versus-control interval included parity. That control can account for the observed B4 partial difference within measurement uncertainty. It cannot rule out small native-branch overhead. Both held-out B1 candidate-versus-control intervals were slightly below parity, although neither candidate-versus-baseline interval excluded parity. The secondary control comparisons do not supersede the formal verdict. [Control comparisons](guarded_aa.json)

### Active real-weight production timings

The held-out values below are synchronized median wall milliseconds for the complete measured context path; the interval uses the paired plugin statistic.

| Batch and commit | Native ms | Candidate ms | Native control ms | Candidate / native |
|---|---:|---:|---:|---:|
| B16_partial | 7.350 | 6.995 | 7.302 | 1.0459 [1.0288, 1.0588] |
| B16_full | 7.270 | 6.928 | 7.289 | 1.0568 [1.0424, 1.0697] |
| B32_partial | 14.961 | 11.787 | 14.868 | 1.2453 [1.2179, 1.2774] |
| B32_full | 14.744 | 11.974 | 15.008 | 1.2273 [1.2089, 1.2429] |

Earlier packaged-primitive CUDA graph checks matched all 20 fresh-input replays on nondefault and alternate streams. The full checkpoint SHA-256 for fc.weight was `036d2e2e85e16a4755f00b234fbe932095949a0d44a4ff7b2e4df821ba959b0a`. The combined verified auxiliary blob SHA-256 was `0aa5203bbc30522ff878f55f7427974004ac42e24c1ae04cd3af2d7e7e4c0309`. Eight-H200 warmup cost, actual complete-model token agreement, TP-wide algorithm consensus, the scheduler's maximum peak memory and serving score were not measured. [Patch](integrated_candidate.diff) · [Held-out details](guarded_holdout_detail.json) · [Source-package graph check](anchor_report.json)
