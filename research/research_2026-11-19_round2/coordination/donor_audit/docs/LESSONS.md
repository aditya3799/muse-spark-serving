# Lessons from two weeks of kernel competition

In rough order of how much they would have saved us had we known them on day one.

## About measuring

1. **Make the test workload identical to the scored one.** Our harness used the default expert range; the judge
   uses each input's busiest range and a router seed tuned to a target load. The MoE did 2.5x less work locally at
   batch 1, which hid a 10 µs gap and mis-measured every MoE A/B on rented pods. Check the generator, the seeds and
   every "case field" the judge fills in.
2. **Read the scoring formula and turn it into a price per microsecond.** Here the score is a geometric mean over
   six loads of weighted time sums, with sliding layers weighted 46 and full layers 16. Saving 1 µs on batch-1
   sliding decode was worth 15 µs on batch-32 full verify. We printed that table into every agent brief.
3. **Know what a profile is measuring.** The judge's per-kernel profile is an eager run on PyTorch's default stream,
   where PDL early launch silently does nothing. A micro-probe showed the same two kernels overlap by 0.4 µs in a
   graph and not at all on the default stream. Our first reading of that profile ("PDL is broken on the judge") was
   wrong.
4. **Noise sets the smallest idea you can verify.** ±1 % per run overall and ±2–5 % per case meant a single check
   could not confirm a 0.5 % change. We designed checks so that two changes touched disjoint cases, or repeated runs.
5. **Expect the winner's curse.** Submissions landed 1–2 % below the check that selected them, because we picked
   the luckiest run.
6. **Check the clock before trusting a pod.** A 630 W pod throttled the expert loop from 1.74 to 0.58 GHz halfway
   through, and that looked like a kernel problem for an evening.

## About performance

7. **At small batch, bytes are the only currency.** Every big win reduced bytes (C12, NVFP4 straight to the tensor
   cores), moved them earlier (PDL weight prefetch), or removed a kernel boundary (fusion). Clever math rarely
   mattered.
8. **Lossless compression is underrated.** Trained BF16 weights live in a narrow exponent range, so 12 bits hold all
   but 1 in 10,000 of them exactly. The rest are patched exactly. That is 25 % fewer bytes with no accuracy question.
9. **A megakernel is not automatically faster.** Our first fusion of the whole front of the layer into one
   persistent kernel lost 9–14 %. Inside one kernel, every stage waited on cross-CTA hand-offs (all-gathers, flags)
   while HBM sat idle. A chain of separate kernels with PDL overlap kept HBM busier. A leaner second version on the
   last day (qkv and attention only, cluster-local hand-offs through distributed shared memory) was 9 µs faster than
   the production front measured alone, yet saved only 0–1.6 µs inside the layer. In a bandwidth-bound layer
   whose kernels already overlap, a shorter front mostly gives its time back to the weight streams behind it.
   Measure the change in the whole layer, not in isolation.
10. **A fork/join in a CUDA graph costs about 3 µs** of replay time even when the side work is fully hidden. One
    stream with fused work usually wins at small batch.
11. **The isolated tuner can be wrong in-layer.** A projection or expert config that won in isolation sometimes lost
    once neighbours overlapped it via PDL. Where it mattered, we pinned in-layer winners
    (for example, the fused expert kernel at 9–32 tokens was 5–10 µs faster in-layer than its isolated tune said).
12. **Instruction count per weight is the expert-kernel lever.** NVFP4 decode at full bandwidth needs about 80 % of
    every SM's issue slots. A lossless K-permutation that saved one scale extraction per MMA, and a one-line
    "warp-uniform" trick that let the compiler keep wgmma descriptors uniform, were each worth more than most new
    kernels.

## About correctness

13. **Persistent kernels need a recheck under fresh inputs and interleaved graphs.** Generation counters, parity
    buffers and early triggers all broke at least once in ways a single test never showed: stale inputs, deadlocks
    and a hang that appeared only when two cases' graphs interleaved.
14. **Audit what the compiler did with your PDL wait.** The compiler moved loop-invariant loads above
    `griddepcontrol.wait` once. The fix is `asm volatile` loads after the wait, and the check is to read the SASS.

## About working with agents

15. **One strong agent with a notebook beats a swarm of weak ones.** Every shipped kernel came from Claude. The local
    8–27B models made 134 attempts and none beat the best version, although the paper scout's idea cards were useful.
16. **Write the brief like you are handing work to a colleague.** Missions that got a goal, a base version, disjoint
    version and job ranges, and the list of dead ends came back with shippable code. Vague ones came back with
    re-measurements of known facts.
17. **The notebook is the memory.** The coordinator's context reset many times. Everything that mattered survived
    because it was written into `notes.md` with numbers, and most good ideas came from re-reading it.
18. **The human's job is judgment.** Where to spend money and judge checks, when to stop tuning and rewrite, and
    when an idea was dead. Those calls mattered more than any single kernel.
