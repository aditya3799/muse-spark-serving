# Compact verification and frozen expert arithmetic

Using the original full-width Humming tile plan at compact M repaired every B16 skew and B32 skew replicate tested on the RTX 3050. B32 uniform still produced 458 differing BF16 results in one of four replicates. Forcing the original expert's path to cross the light/heavy cutoff changed 512 outputs in two B32 trials. Supplying the original dense route histogram recovered exact output in all diagnostic cases. [Tile plan](pin_held.json) · [Forced crossing](pin_crossing.json) · [Oracle](oracle_suite.json)

For retained count c and d omitted rows, the unknown original count can range from c to c+d because top-k contains each expert at most once per row. If this interval crosses the cutoff and the two kernels reduce K differently, the compact input alone cannot determine which answer the dense run produced. Every SM90 batch contained such a route in 16 CPU draws per load and distribution using the held-out width masks and the pinned heuristic tables. The B32 GPU counterexample demonstrates this ambiguity can affect a concrete output. [CPU guard](guard.json)

| RTX expert-block shape | Median original / compact | Simulated commits retained | BF16 differences in four rounds |
|---|---:|---:|---:|
| B16 uniform | 1.182× | 85.2% | 0 |
| B16 skew | 1.713× | 85.2% | 0 |
| B32 uniform | 1.107× | 77.4% | 458 |
| B32 skew | 1.321× | 77.4% | 0 |

These measurements cover one rank's W13, activation, W2, finalization and compact row packing, on synthetic tensors. Commit ratios come from simulated observations of public GLM acceptance rates. No TP8 serving or Muse token-agreement measurement is available.