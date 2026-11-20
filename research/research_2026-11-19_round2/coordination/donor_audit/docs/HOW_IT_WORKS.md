# How it works: making one MoE decoder layer 2x faster than SGLang

This is the engineering story behind `solution/decode_layer.py`: what the layer computes, why its speed is a
memory problem, and every optimization that moved the judge score from 1.06x to 2.06x. The numbers are
judge results unless they say "pod" (a rented H200) or "5090" (the local GPU).

## 1. The problem

One decoder layer of Muse Spark Open, as one GPU of an 8-GPU deployment (tensor-parallel 8, expert-parallel 8)
sees it. The cross-GPU all-reduces are left out:

```
x: attention input [T, 8192] bf16      s: residual stream fp32
q, k, v, g = split(x @ qkv.T)          QK-norm, RoPE (sliding layers), write fp8 k, v into a paged cache
a = (rmsnorm(attention(q)) * sigmoid(g)) @ o.T
s = a1*s + b1*rmsnorm(a);  h = rmsnorm(s) * (1 + w)
experts = top8(sigmoid(0.177 * h @ router.T) + bias)          # 256 experts, 32 live on this GPU
y = local_experts(rmsnorm(h @ pooler.T) * (1 + w))            # NVFP4 weights, bf16 activations
f = y @ unpooler.T
s = a2*s + b2*rmsnorm(f * (1 + w));  return rmsnorm(s) * (1 + w), s, experts
```

It is timed on 12 shapes: decode (1 token per request) and verify (an 8-token speculative draft per request), at
batch 1, 16 and 32, each on a sliding-window layer (2,048-token window) and a full-attention layer (80,000-token
context, FP8 KV cache with one token per page). That is T = 1 to 256 tokens per call. Each load's time is
`46 x sliding + 16 x full`, and the score is the geometric mean over the six loads of SGLang's time divided by ours.
Every call is a CUDA-graph replay with a cold L2 cache, every output is checked against an FP32 reference, and the
judge picks a router seed so each load gets a realistic amount of expert work (about 2.5 active local experts at 1
token, 7.5 at 8–16, 9 at 32, 18.5 at 128 and 21 at 256).

## 2. Why it is a memory problem

Per call the layer reads the dense weights once: qkv 37.7 MB, o 16.8 MB, pooler 67 MB, router 8.4 MB, and the
unpooler (8.4 MB slice up to 8 tokens, 67 MB above). That is 140–200 MB. On top of that come the active experts at
about 28 MB each in NVFP4, plus the attention's KV reads. With T ≤ 32 tokens every weight byte feeds at most 32
multiply-adds, far below what the tensor cores need to be the bottleneck. So the floor is bytes / bandwidth: 40 µs at
batch-1 decode, 85 µs at batch-32 decode (the judge reports this "speed of light" per case).

Three things stand between a kernel and that floor:

1. **Bytes in flight.** Each H200 SM needs tens of KB of outstanding loads to cover HBM latency. GEMV-shaped kernels
   that issue one load per warp and wait never get there.
2. **Gaps between kernels.** A kernel boundary in a CUDA graph costs 1–1.5 µs, plus the ramp-up and tail of each
   kernel. Our first version ran about 20 kernels at batch 1, so up to ~20 % of the time was boundaries.
3. **Decode work on the CUDA cores.** NVFP4 expert weights must be turned into BF16 tensor-core operands, which costs
   about 2.4 instructions per weight. At 4.3 TB/s that keeps every SM's issue slots very busy.

## 3. The path from 1.06x to 2.06x

### 3.1 Start from SGLang, tune everything (Oct 23–26: 1.06x → 1.54x)

The first versions kept SGLang's structure and replaced its generic choices with tuned ones:

- **Per-shape autotuning at prepare time.** The judge gives 300 s for `prepare()`. We use them to time
  candidate configurations of every projection (Triton, cuBLAS and pinned cuBLASLt algorithms, later our own engines)
  and every expert kernel, at each token count, on the judge's own GPU, then pick the fastest. The CUDA modules
  compile in parallel threads while the tuner runs.
- **Glue removal.** A torch layer-norm plus a local-ids kernel became one Triton kernel. Split-K projections
  stopped needing a zero-fill kernel and a cast kernel (the last split to arrive sums the partials and writes bf16),
  and the expert kernels were tuned per token count. decode_v23: 20 → 17 kernels at batch 1, +3.2 % on the same pod.

### 3.2 Attention (Oct 25–Nov 2)

- **Ring attention for the sliding window.** A cluster of CTAs splits the 2,048-key window. Each CTA dequantizes its
  FP8 keys and values, and the partial softmax results are combined through distributed shared memory in a fixed
  order. Pod A/B, ring on vs off: decode-1 −3.8 %, decode-16 −1.9 %, decode-32 −1.0 % time.
- **Full-attention decode** over 80,000 keys: dedicated kernels for 1 request (`b1`) and for 2–32 requests
  (`fa2`, warp-specialized and key-balanced so every SM gets the same number of keys).
- **Verify (8 query tokens).** `fa8` handles all 8 draft tokens of a request in one pass over the keys
  (QK^T with wgmma, softmax straight into FP8 codes, FP8 P·V). The prepare-time tune measured 107 vs 124 µs at batch
  16 and 195 vs 227 µs at batch 32 against our previous kernel; judged together with other changes as decode_v347
  (2.03x check).

### 3.3 C12: lossless 12-bit weights (Oct 29–30)

The dense projections are pure weight streaming, so the cheapest way to make them faster is to read fewer bytes,
**without changing a single bit of the result**.

- **The format.** A BF16 weight is 1 sign, 8 exponent and 7 mantissa bits. The weights of one trained layer sit in a
  narrow exponent range, so C12 keeps the sign, a **4-bit exponent offset** inside a per-layer window of 16
  binades, and the 7-bit mantissa: 12 bits instead of 16. Values outside the window (about 1 in 10,000–25,000)
  are stored as exact exception records (row, k, correction) and added back in the epilogue. Decoding is bit-exact.
- **The decode trick.** The 12-bit codes sit in place in the BF16 bit layout, so one `LOP3` per register
  (`(w & 0x87FF87FF) | 0x38003800`) turns two codes into two valid BF16 numbers whose exponent is offset by a
  constant. The whole tile is then rescaled by a power of two once, at the end. The weights go from global memory
  into registers and straight into the MMA, with no shared-memory round trip.
- **Engines.** `c12m` (1–8 tokens, mma.sync, weights in registers), `c12r` (2–32 tokens, TMA ring, decoded into
  wgmma's A operand), and fused variants: `c12b` (o-projection or unpooler plus the norm/residual boundary) and
  `c12pr` (pooler plus router in one pass over the same activations).
- **Effect.** 25 % fewer projection bytes. In the prepare-time tune at 1 token: qkv 11.0 vs 12.2 µs, pooler 15.5 vs
  18.3 µs, against our best BF16 kernels.

### 3.4 Our own NVFP4 expert kernels (Oct 24–Nov 3)

The experts are the largest single cost. At batch-32 decode they take half the layer time.

- **One persistent launch per MoE call**, grid = number of SMs, with warp specialization: a producer warp streams
  weight stages into a shared-memory ring with TMA bulk copies, consumer warps decode NVFP4 (4-bit E2M1 plus an FP8
  scale per 16 weights) into BF16 tensor-core operands, and an epilogue warp applies SiLU-GLU and the weighted sum.
- **A kernel per regime.** `ma` at 1 token (see 3.6), `m4` at 2–16 tokens (CTAs form a team per expert job),
  `moe_fk` (gate/up and down fused) and `moe_fw`/`moe_mx` at 17–256 tokens (register-sourced wgmma: the decoded
  weights never touch shared memory).
- **Instruction diet.** The decode is the hot loop, so every instruction counts:
  - *KPERM.* A lossless K-permutation of the weights, applied at prepare, so each lane needs one scale pair per 64-k
    chunk instead of one per MMA. It cut the byte-permute instructions in the kernel from 900 to 228 (20.7 → 19.4
    instructions per 16-k step).
  - *The warp-uniform trick* (`warp = __shfl_sync(~0u, threadIdx.x >> 5, 0)`). It lets the compiler keep wgmma
    descriptors on the uniform datapath: moe_fw went from 292 to 259 µs at 256 tokens. Judged together with a new
    batch-1 attention kernel as decode_v94: 1.80x → 1.88x.
- **m4 at 2–16 tokens** (decode_v420): one launch that does pre-expert norm, local ids, teams of CTAs per
  (expert, ≤ 8-token) job, cross-phase prefetch of the down-projection weights, and the weighted sum. +4 % at
  batch-16 sliding decode.

### 3.5 Overlap: programmatic dependent launch everywhere (Oct 24–Nov 1)

With PDL, kernel N+1 can be scheduled while kernel N is still running. Each of our kernels issues its **weight**
loads first, then `griddepcontrol.wait` (wait for the previous kernel's results), then reads its activations. So
each kernel's weight stream starts during its predecessor's tail. The timeline in the README shows the result: the
pooler starts streaming while the o-projection is still running, and the unpooler while the MoE finishes.

- decode_v99 (PDL for the expert kernels) +0.8 %, v104–v106 (PDL at 8 to 32 tokens) +0.9 %.
- The rule that made it safe: every read of a predecessor's output must sit after the wait. We found one case where
  the compiler hoisted a load above `griddepcontrol.wait` by reading the SASS, and fixed it with `asm volatile`
  loads. Every new kernel was audited the same way.

### 3.6 Fusion and the one-launch MoE (Nov 1–3)

- **Boundaries fused into their neighbours.** The norm-plus-residual boundary runs inside the o-projection (`c12b`
  at 9–32 tokens) or as a cluster kernel (`nr_k`). The unpooler and the post-FFN boundary are one kernel at 1–8
  tokens (`ub8`, with a single grid barrier) and at 16–32 tokens (`c12b`).
- **`ma`: the whole FFN up to the experts in one persistent kernel at 1 token.** It does the router logits, top-8
  routing, the 67 MB pooler GEMV, the pre-expert norm and every routed expert. The expert weights start streaming as
  soon as routing finishes, while the pooler is still running. Judge profile at batch-1 decode: 6 kernels per layer
  in total (qkv, attention, o-projection, boundary, `ma`, unpooler+boundary).

### 3.7 Engineering that kept it all working

- **One 1 MB file.** The submission must be one Python file under 1,000,000 bytes, yet it carries 25 CUDA modules.
  `tools/build/slim.py` strips comments and whitespace from the Python and the embedded CUDA. Every version is
  produced by builder scripts applied in sequence (`tools/build/merge_builders.py`).
- **A prepare that survives the judge.** All modules compile in parallel threads with a deadline. When the judge
  environment changed mid-contest (new driver, new config) and prepare ran out of time, unfinished tunes fell back
  to known-best defaults (decode_v175).
- **Correctness under fresh inputs.** The judge replays one graph on new inputs every call, so every
  generation-counter or parity scheme (persistent kernels reuse their scratch across calls) was stress-tested with
  thousands of interleaved replays on the 5090.

### 3.8 The last day: in-layer tuning and a second megakernel (Nov 4)

- **In-layer re-ranking of the tuner** (decode_v480–v482). The prepare-time tuner times each projection in isolation,
  but neighbours overlap it through PDL in the real layer. The final version keeps the tuner's five best validated
  candidates per (projection, token count) and re-times them inside the whole layer before choosing. It mostly
  confirmed the isolated pick, except at 32 tokens (qkv −1.4 µs, pooler −2.7 to −4.7 µs, fused o-projection boundary
  about −2 µs) and for the fused unpooler boundary at 16 tokens (about −2 µs). decode_v482: 2.0571 check, **2.0561
  submitted**, our best judged score.
- **A second megakernel for the 1-token front** (decode_v483–v510, `kernels/experimental/megakernel/`). One persistent
  launch of 120 CTAs in clusters of 8 does the qkv projection (C12, every CTA streaming a share), QK-norm, RoPE, the
  KV-cache write, sliding-window attention and the gate. The 64 attention CTAs build their heads' q from per-CTA
  partial sums through distributed shared memory. Values cross CTAs as {value, tag} 64-bit words, so a reader can tell
  a fresh word from last call's without a separate flag. The kernel stops after the attention and hands off to the
  production o projection, which starts streaming its weights through PDL while the attention still runs. Measured
  alone with a cold L2 (each followed by the same o projection and boundary), it beat the production front 28.8 to
  38.0 µs. Inside the layer it saved only 0 to 1.6 µs at single-request sliding decode, about 0.2 % of the score and
  inside the judge's noise. Its check scored 2.0579 and its submission 2.0301. The production front was already hidden
  behind the weight streams of the kernels that follow it, so making it shorter barely moved the end of the layer (see
  `docs/LESSONS.md`).

## 4. Where the remaining time goes

Compared to the speed of light, the final submission is at 1.5–2.3x the floor depending on the case (see
`docs/img/per_case.png`). The judge's own per-kernel profiles show the gap is mostly in two places:

- **The expert kernels run at 2.5–3 TB/s, not 4.3.** At 8–16 tokens `m4` takes about 79 µs standalone where the
  bytes alone would take about 50 µs. The losses are the start (routing tables before the first weight lands, about
  4–7 µs), the transitions between the gate/up and down phases, and the slowest team setting the end.
- **The front of the layer at 16–32 tokens is a latency chain**: qkv → attention → o → pooler → router → routing,
  each waiting on the previous one, with about 20 µs of it latency rather than bytes.

The ideas we tried against both, and why most of them did not survive the judge, are in `notebook/` and
`docs/LESSONS.md`.
