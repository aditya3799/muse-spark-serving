# Frozen-parent FFN time by stage

On the RTX 3050, W13 and W2 together used 80.1–90.1% of the measured rank-local FFN. W13 used 55.2–60.2%. The table gives paired median fractions of the marked GPU stream time and independent uninstrumented GPU replay medians. A group of independently calculated median fractions need not sum to 100%.

| Rows / route fixture | Local pairs / active experts | Plain FFN ms | W13 | W2 | W13 scaling + SiLU | Post-expert finalize | Dense projections | Marked / plain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 normal | 267 / 17 | 5.68 | 60.2% | 29.9% | 2.3% | 0.5% | 4.9% | 1.036 |
| 128 skewed | 239 / 3 | 15.47 | 55.2% | 24.9% | 6.1% | 1.1% | 10.1% | 1.027 |
| 768 normal | 1,233 / 28 | 17.95 | 58.1% | 28.9% | 3.3% | 0.8% | 8.4% | 1.005 |
| 768 skewed | 2,520 / 6 | 20.94 | 56.1% | 28.6% | 5.7% | 1.0% | 7.2% | 1.004 |
| 16,384 normal | 30,973 / 29 | 247.59 | 56.0% | 27.4% | 5.6% | 1.2% | 8.3% | 1.047 |
| 16,384 skewed | 45,235 / 3 | 371.50 | 57.7% | 28.5% | 5.8% | 1.1% | 6.1% | 1.002 |

At the measured stage fractions, halving both expert GEMM times while holding every other cost fixed would imply 1.67–1.82× faster execution of this local FFN call. This is an arithmetic sensitivity. It cannot establish a H200 serving score.

Each input used the kernel task’s own `generate_input(..., sharding="ep8", rank="busiest")` routes and top-k scores. The serving workload’s expert IDs were unavailable. Full 32-expert histograms, route fixture seeds, plain synchronized host-wall timings, individual stage timings and all samples are in the [raw receipt](timings.json).

For M128 and M768, eight pairs alternated marked CUDA graph replays and separately captured uninstrumented CUDA graphs. M16384 ran six pairs with serial, unchunked eager calls per fixture. Each measured arm retained identical finalized BF16 outputs and used the scored Humming helper. The sum of timed components was 0.1–2.2% above independent uninstrumented time in the paired median. Marked full-call time rose by 0.2–4.7%. M128 normal includes a 17.3× longest-to-shortest spread in plain times. The benchmark omits peer communication and router/attention work. [Summary](summary.json) · [Reproducible harness](paired.py)

