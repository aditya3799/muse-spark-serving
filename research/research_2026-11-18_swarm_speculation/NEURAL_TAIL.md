# Extending the intact DFlash chain

The policy retains the trained DFlash anchor and seven neural draft IDs, looks up that eight-token key in the request's confirmed history, and appends 24 historical tokens when available. The source index contains a key only after its entire continuation is in the past within one message. The experiment uses reconstructed public trajectories and the exact Muse tokenizer. All real neural draft logits and per-step hardware timings are unavailable.

The baseline and each policy share deterministic simulated neural predictions and greedy targets. Partial acceptance varies across one through seven commits. Cases with 0%, 10%, 25% and 40% actual full-eight commits were each calibrated to approximately four emitted tokens per baseline verify, including final turn truncation. The correlation sensitivity multiplies the odds of a full-eight baseline commit by four when the underlying correct eight-token chain has a historical source. This defines a simulated dependence, and the serving lookup sees only the proposed eight IDs. A concrete partial chain takes the correct prefix, diverges at its first rejection and reuses a known historical alternative if one exists. Additional cases cover deterministic perturbed candidates and an upper bound charging every budget-eligible partial rejection a wide verify.

| Corpus | Generated tokens | 0% full | 10% full | 25% full | 40% full |
| --- | ---: | ---: | ---: | ---: | ---: |
| SWE-smith, 40 held-out trajectories | 216,069 | 0.977× | 1.109× | 1.223× | 1.271× |
| OpenHands, 491 trajectories | 1,256,669 | 0.980× | 1.044× | 1.108× | 1.134× |
| Muse-generated coding, 20 trajectories | 914,060 | 0.968× | 0.996× | 1.014× | 1.004× |

These ratios compare single-request decode work against the eight-token baseline. They use independent full acceptance and charge 1.25 baseline steps for each 32-wide step, with the neural draft still performed. Positive correlation on the Muse trajectories changes the 10%, 25%, and 40% values to 1.012×, 1.014× and 0.985×. The independent 25% case needs 1.616× the target token rows for 1.077× fewer target calls. The wide-step cost must stay below 1.311 baseline steps to break even. Of 52,329 wide verifies, 43,426 reject inside the original seven neural choices. The trace-bootstrap ratio interval at the 1.25× cost assumption is 0.994–1.034. Single-request replay omits batch padding, prefill, router overhead, numerical variation across GPU shapes and the official score calculation.

The independent greedy serial test passed 5,120 early and late rejection comparisons and four causal-source checks. Under identical target predictions, the extended chain emits every accepted baseline token and exactly one target bonus. The target changes neither the original accepted prefix nor the next token on any early rejection. Source selection can still waste a wider target call. The pinned request protocol exposes `spec_correct_drafts_histogram`; index 7 divided by its sum measures the checkpoint's full-seven draft acceptance when collected.

The observed work counts on actual Muse-generated text do not justify integrating or submitting this branch. Raw per-case target counts, row inflation, assumed-cost sensitivities and trace bootstrap intervals are in the three `neural_tail_*.json` reports.

