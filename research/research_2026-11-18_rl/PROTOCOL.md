# Local serving benchmark

The reference is the full release patch used by judge job `317c88a5fd474cd2`, SHA-256 `8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`, applied to SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90` with `--moe-runner-backend humming`. Its measured eight-H200 score is 1.2363×. All candidate records use this release as their parent through November 19 as requested.

## Reproduce the reward

From the serving directory, use Python with NumPy and the kernel-autoresearch-plugin checkout at `../scratch/kar-serving-research`, revision `64d53f37009e46ae9a05c8c319d44a90f9740bb3`.

```
python research_2026-11-18_rl/test_rewards.py
python research_2026-11-18_rl/score_trials.py research_2026-11-18_rl_prefill/pilot.json --holdout research_2026-11-18_rl_prefill/holdout.json
```

For a new GPU candidate, use the WSL driver to acquire the local lock, run both seeds, verify the release hash, compute rewards and update the ledger. Use a descriptive run name and the adapter that executes the candidate. Repeating a name creates a timestamped directory. When available, the driver records clocks, power, temperature and GPU memory once a second. For example:

```
python research_2026-11-18_rl/run_pair.py --name first_turn_recheck --seeds 17355 88419 -- python research_2026-11-18_rl/bench_long_aux.py --seed {seed} --out {output}
```

Each agent records its name, patch or variant, timed operation, hardware, frozen parent hash, input seed, correctness, dispatch evidence and individual samples. Pair each candidate with the parent on the same GPU, shape, routes, inputs, capture mode and clock conditions. Use an even number of measurements with alternating AB and BA order. Interleave the unchanged parent as a third arm. Give each agent exclusive GPU time. A confirmation changes the input seed and uses the same shape set, weights and candidate. Match the measured operation before comparing absolute times across agents.

The plugin computes geometric speedups in log space, balances order and bootstraps paired measurements. Each named shape gets equal weight unless an experiment documents workload-specific weights up front. Retain timings, confidence intervals and decisions by shape. The signed reward for a resolved improvement or regression is `100 × ln(speedup)`. Inconclusive results or noisy controls earn zero. Mixed results receive the worst confirmed shape regression as a penalty. Incorrect output earns −100. Unsupported dispatches and incomplete trials have no performance reward. A local promotion requires two independent controlled KEEP verdicts. Rewards across different operations measure different quantities and cannot be added.

## Local coverage

The local RTX 3050 has 6 GB and one rank. Tests cover release code that fits on it using model dimensions and declared route distributions. State which operands are synthetic and record source revisions and hashes for checkpoint weights. The public DFlash2 tests use true weights with independently generated target activations. Record whether a measurement covers one operation, both expert matrix multiplies and their finalization, or a wider serving step. For a chunked replay, report the actual chunk size, local routing distribution and whether copies and finalization were timed. Record local peak allocation when memory limits require adapting a workload.

A change to measured kernel time carries a score projection only with matched evidence for its share of serving latency, distributed communication, token agreement and each load. The four scoring concurrencies are 1, 4, 16 and 32. Keep the official 1.2363× number and local relative ratios in separate fields. Ask the user before any scored submission.

Sources include the [paired benchmark statistics](https://github.com/gauravjain14/kernel-autoresearch-plugin/blob/64d53f37009e46ae9a05c8c319d44a90f9740bb3/kar/evaluate/stats.py), [public Hopper kernel measurements](https://github.com/alygg77/okc-muse-spark-kernels), [pinned SGLang release](https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90) and the [serving rules](https://openkernelchallenge.dev/rules).