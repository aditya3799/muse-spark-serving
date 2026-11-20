# Kernels

`decode_layer/` holds the 25 CUDA modules embedded in `solution/decode_layer.py`, extracted verbatim from the
annotated source (`solution/decode_layer_annotated.py`), one file per module. `prepare()` compiles them with
`torch.utils.cpp_extension.load_inline` and a minimal header shim (`lite_header.cu`) instead of the full torch
headers, which keeps compile time inside the judge's prepare budget. Hopper-only modules (wgmma, TMA, clusters)
are compiled for `sm_90a`.

| File | Kernel(s) | Used for |
|---|---|---|
| `c12m_cuda.cu` | `c12m_k` | C12 (lossless 12-bit) projections at 1–8 tokens: weights global → registers → mma.sync, optional cluster split-K |
| `c12r_cuda.cu` | `c12r_k` (+ variants) | C12 projections at 2–32 tokens: TMA ring, decoded into wgmma's A operand (pooler, unpooler, o) |
| `ub_cuda.cu` | `c12b_k` | C12 projection fused with the norm/residual boundary (o + post-attention boundary, unpooler + post-FFN boundary) at 9–32 tokens |
| `c12pr_kern.cu` | `c12pr_k` | Pooler and router in one pass over the same activations (2–8 tokens) |
| `gv_cuda.cu` | `gv_*` | 1-token row GEMVs (BF16 and C12) |
| `tg_cuda.cu` | `wgs5` | TMA + wgmma streaming GEMM over pre-tiled BF16 weights (qkv at 16–32 tokens) |
| `ring_cuda.cu` | `sl::attn4_k` and helpers | Sliding-window decode attention split across a CTA cluster, FP8 KV, DSMEM combine |
| `b1_cuda.cu` | `b1_attn` (+ variants) | Full-attention decode, 1 request |
| `mbf_cuda.cu` | `fa2_k` | Full-attention decode, 2–32 requests (warp-specialized, key-balanced) |
| `fa8_cuda.cu` | `fa8_k`, `fa8c_k` | 8-token verify over full attention, FP8 tensor-core P·V |
| `l8b_cuda.cu` | `attn19_k`, `attn16_k` | Verify attention: `attn19_k` for 1 request (full layer), `attn16_k` for sliding verify at 16–32 requests |
| `fd2_cuda.cu` | split-KV decode/verify | Older full-attention path (kept as a fallback) |
| `nr_cuda.cu` | `nr_k` | Norm + residual boundary as a cluster kernel |
| `ub8_cuda.cu` | `ub8_k` | Unpooler slice + post-FFN boundary in one kernel at 1–8 tokens (one grid barrier) |
| `u8k_cuda.cu` | `u16` | Unpooler slice alone at 1–8 tokens (fallback) |
| `mak_cuda.cu` | `ma_k` | **One persistent kernel for router, routing, pooler, pre-norm and all experts** at 1–8 tokens |
| `m4_cuda.cu` | `m4_k` | NVFP4 experts at 2–16 tokens: teams of CTAs per expert job, in-kernel routing at 2–8 |
| `fk_cuda.cu` | `moe_fk` | NVFP4 experts, gate/up + down fused (≤ 32 tokens) |
| `fw_cuda.cu`, `mx_cuda.cu` | `moe_fw` | NVFP4 experts at 17–256 tokens, register-sourced wgmma, stream-K units, KPERM layout |
| `stream_cuda.cu` | `moe_stream`, `moe_sk` | TMA streaming / stream-K experts (earlier generation, still tuner candidates) |
| `mma_cuda.cu` | `moe_mma2` | mma.sync experts (a tuner candidate) and the weight packing the other expert kernels reuse |
| `r32_cuda.cu` | `r32_k` (+ variants) | FP32 router GEMV, optionally with routing, at small token counts |
| `px_cuda.cu` | `px_k` | Route lists + pre-expert norm in the permuted order (older MoE path) |
| `lite_header.cu` | — | `okc_lite`: the minimal torch-API shim for the extension glue |

`experimental/m4_steal.cu` is `m4` with cross-team work stealing: when a team runs out of units in a phase, its
producer claims units of the job with the most left. It was correct on every test but not shipped, because the
pod measurements that rejected it turned out to use a lighter MoE load than the judge (see `docs/TESTING.md`).

`experimental/megakernel/` is the one-launch front of the layer at 1 token (decode_v483–v510): qkv (C12), QK-norm,
RoPE, the KV-cache write, sliding-window attention and the gate in one persistent kernel of 120 CTAs in clusters of 8,
followed by the production o projection. `mkf_cuda.cu` is the kernel, `mkf_glue.py` the host side (exception blocks,
the in-prepare correctness check and in-layer timing against the production front), and `build_decode_mkf.py` the
builder that adds it to a base version (paths are the lab layout). `tools/probes/mkf_test.py` is the 5090 test driver.
It is exact against the reference and switched itself on in the judge's in-layer timing, but its gain (0–1.6 µs at
single-request sliding decode) is inside the noise; see `docs/HOW_IT_WORKS.md`, section 3.8.
