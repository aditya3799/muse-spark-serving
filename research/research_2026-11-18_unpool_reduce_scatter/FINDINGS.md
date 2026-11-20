# Reduce-scatter before the Muse Spark unpool projection

The candidate changes the eligible Humming/Marlin tensor-parallel branch in the scored 1.2363x serving patch (job 317c88a5fd474cd2). Each rank now reduces its contiguous expert-output rows, applies the same full-K matrix multiplication and BF16 conversion as the scored implementation, and gathers the projected rows in the original rank order. The scored and candidate implementations share the local projection routine. Both transport operations use the existing TP group.

The isolated combined candidate is `candidate.diff` (SHA256 7f6826075c0207c783586c356886b2a6b7a276c5fab354019fa7978bdf931f34). `delta.diff` is the two-file change relative to the scored patch. `build.py` authenticates its parent, reconstructs the pinned files, applies and verifies both patches, and compares the resulting two files byte-for-byte with the generated sources.

## Qualification

Fourteen dispatch cases exercised the extracted candidate model methods. The matrix covers divisible rows, the nondivisible 753-row tail, small batches, another dtype, strided inputs, already-reduced output, expert TP, SP, attention DP and CP, deterministic inference, existing multicast ownership, and missing projection weights. The candidate reuses the scored Hopper, contiguous-weight, width/alignment and row eligibility checks. The new contiguous-input check protects the collective input contract.

Actual BF16 Gloo all-reduce and reduce-scatter followed by row all-gather gave bitwise-equal results at 2, 4, and 8 ranks. Each configuration compared 934,272 projected BF16 values across integer, random, sparse and cancellation fixtures, including 16,384 rows. The CPU fixture computes each BF16-input dot product in FP32 and casts the result to BF16. Its 64-by-48 weight makes the result a decomposition and transport test. It does not exercise SGLang's CUDA group dispatcher, CUDA graph capture, full model widths or NCCL. The per-rank JSON files retain the raw counts.

A separate order-sensitivity experiment changed only the BF16 accumulation tree. Left-to-right versus balanced summation changed the final projection by 0.50% relative L2 on seeded dense inputs, 0.41% on sparse inputs, and 32.94% on an adversarial cancellation fixture. These are synthetic alternatives. The actual NCCL protocols, their accumulators, and token agreement on H200 remain unmeasured. The earlier custom two-shot proof reuses the original reduction order but its 16 MiB limit misses full 16,384-row prefills. This new group-based path can address those large shapes. The archived experiments contained no matching eight-H200 timing for the replacement collective.

## Critical-path relevance

The official component reference specifies a 4096-wide expert output and an 8192-wide model stream. A TP8 logical ring model sends 448 MiB per rank for all-reduce plus projected-row all-gather at 16,384 BF16 rows. Reduce-scatter plus the same gather sends 336 MiB. The difference is 112 MiB per layer per rank, a 25% reduction across this pair. The new rank-local preprojection buffer occupies 16 MiB. A 753-row or 3012-row input takes the scored path at TP8. Padding would change the parent GEMM geometry and the transport/compute balance; this candidate keeps these inputs unchanged.

Even if equivalent bandwidth made collective latency scale exactly with these byte counts, the pair alone would improve 1.333x; GEMM, expert execution, attention and the other per-request work retain their costs. No measured serving speedup or claim of 1.8x follows. This candidate has not been submitted.

Run `python build.py`, then `python qualify.py --ranks 2`, `--ranks 4`, and `--ranks 8` using a PyTorch installation with Gloo. Run `python analyze.py` to reproduce the logical traffic and reduction-order bounds. The observed CPU tests used PyTorch 2.6.0+cu124 on Windows.

