# Combined scale-order and group-8 activation candidate

The combined patch improved rank-local FFN time by 0.54% on the held-out seed across the twelve equally weighted verification and prefill cases. The paired interval was 0.13–0.95%. Every finalized BF16 output matched the scored parent and the fused-only arm. Both independent FFN runs earned KEEP against the parent. The component increments do not establish a score improvement on H200.

| Paired comparison | Scope and cases | Pilot gain, interval | Holdout gain, interval |
|---|---|---:|---:|
| Combined / scored | Complete routed MoE, 12 cases | 1.0264× [1.0156, 1.0406] | 1.0102× [1.0036, 1.0163] |
| Combined / scored | Rank-local FFN, 12 cases | 1.0109× [1.0076, 1.0138] | 1.0054× [1.0013, 1.0095] |
| Combined / fused-only | Complete routed MoE, four prefill cases | 0.9501× [0.8384, 1.0155] | 1.0358× [1.0074, 1.0811] |
| Combined / fused-only | Rank-local FFN, four prefill cases | 1.0114× [1.0054, 1.0180] | 1.0158× [0.9975, 1.0355] |

**The extra prefill gain from scale order is unresolved.** For the fused-only comparison the plugin marked the FFN trials NOISY and NO-GAIN, and the complete-MoE trials NO-GAIN and KEEP. The M768 uniform MoE pilot contains a ninefold spread between the longest and shortest combined events, and the holdout has a 2.7× spread on each arm. All samples are retained. At M768 the held-out combined FFN took 11.26 ms instead of 11.65 ms on uniform routes and 34.84 ms instead of 35.81 ms on clustered routes. M3072 reversed direction across seeds. Held-out medians were 36.88 ms against 36.47 ms for uniform and 118.83 ms against 118.44 ms for clustered. The aggregate includes both regimes.

**Patch and dispatch.** The cumulative patch SHA256 is `72fd46dfd10c4db0d2e25602673bfb5c40752f9da9729cfca24fdaf1b28ad426`. It combines the scale-order implementation attributed to Humming PR 107 with the independently tested group-8 scaled activation. A private temporary Git index replayed the scored parent, both delta and cumulative patches, and verified the same resulting tree. Each seed identified 24 registered Humming reference and candidate cubin entries. The candidate invoked both specialized W13/W2 kernels at every geometry. Both group-8 arms used stock scaling and SiLU at M8/32/128/256 and actually dispatched the group-8 native activation at M768/3072. The primary two-input implementations match the individually verified source bytes. [Composition](composition.json) · [Candidate](candidate.diff) · [Upstream scale-order source](https://github.com/vllm-project/humming/pull/107)

**Measurement.** One RTX 3050 Laptop 6 GB executed three independent captured graphs in alternating order on uniform and clustered routing fixtures. Each round replayed duplicate parent and fused-only references. All three graphs shared identical routes, weights and activations including all 254 finite signed E4M3 values per expert. There were eight paired measurements for each of 24 scope and route combinations per seed. All 384 finalized output comparisons passed. The FFN includes the rank-local 8192→4096 projection, pre-expert RMS, complete W13/activation/W2 and score finalization, and the local 4096→8192 projection. The other ranks' pool inputs were staged outside timing and the expert reduction used the local partial. Attention, the router and cross-rank collectives were outside the measurement. [Full pilot](pilot.json) · [Full holdout](holdout.json) · [Per-shape medians](summary.json)

[Original-reference MoE results](moe_parent_reward.json) · [Original-reference FFN results](ffn_parent_reward.json) · [Fused-reference MoE results](moe_ballot_reward.json) · [Fused-reference FFN results](ffn_ballot_reward.json)

