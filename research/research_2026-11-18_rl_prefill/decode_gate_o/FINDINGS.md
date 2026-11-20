# Fused attention output gate and row-parallel O projection

The full-width, single-launch SM86 candidate failed the local screen. Across five independent patterns per shape its gate matched the pinned BF16 JIT on every bit. The projection changed 20 BF16 outputs at one row. The four larger shapes preserved all bits on the fixtures and changed-input CUDA graph replays, but ran 23% to 29% slower. Exclude this variant from a scoring package.

| Rows | Output differences across the five static fixtures | Parent / candidate median stage duration |
|---:|---:|---:|
| 1 | 6, 3, 0, 6, 5 | Timing skipped |
| 4 | 0, 0, 0, 0, 0 | 0.81× |
| 8 | 0, 0, 0, 0, 0 | 0.79× |
| 13 | 0, 0, 0, 0, 0 | 0.79× |
| 16 | 0, 0, 0, 0, 0 | 0.78× |

The direct diagnostic fed our gated BF16 input back into the original PyTorch F.linear. It reproduced every discrepancy relative to the new projection, localizing the difference to the custom matrix accumulation. Each timed shape has 10 balanced CUDA graph measurements at 40 replays each and a native baseline-copy control. The native controls agreed within 0.3% in median. Seed 6317321, including cancellation, very small finite inputs and saturated sigmoid. Raw results are in [pilot.json](pilot.json). The rejection stopped the experiment before confirmation.

## Computation and integration scope

The active Muse decode path applies the native FusedRMSNormGate JIT after paged attention. It uses head RMS and sigmoid in a single launch. The ordinary unquantized row-parallel O then uses F.linear to produce a BF16 partial of width 8192 before the existing tensor-parallel reduction. After this reduction the scored model has one fused post-norm, FP32 residual update and pre-FFN norm, with router input when applicable. The candidate computes the gate from the actual strided QKV view, rounds to BF16, accumulates a 1024-by-8192 projection in FP32 with WMMA, and returns BF16. Its isolated hook targets the original collective and caller ownership. The benchmark exercised the pinned native JIT, original PyTorch projection and candidate full projection on one local rank. The collective, later residuals, SGLang serving graphs and H200 were outside the trial. [CPU source and geometry qualification](cpu.json) and [offline SM86/SM90 builds](offline.json).

The rank-local attention input already combines Q, K, V and the O gate in one weight multiplication. The layers after the first already produce their input norm inside the previous FFN boundary. The existing projected [Q/K-RMS, rotary and KV-store candidate](../decode_qkv/FINDINGS.md) handles the other supported entry fusion. Output RMS depends on the result of paged attention. Moving only sigmoid to this earlier entry would insert an extra same-precision FP32 buffer and retain both launches. Public FA3 handles split combining inside its host entry. Gating partial softmax numerators before combining would change the function and rounding. [FA3 source](https://github.com/sgl-project/sgl-flash-attn/blob/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp).

The vendor custom-allreduce algorithm also changes with tensor size and topology. Combining its BF16 rank-order reduction with the learned Muse residual, the two rounding boundaries and optional router output would require a separate multi-rank implementation and validation. The output projection trial leaves this stage alone.
