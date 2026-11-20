# Unweight Kernels

CUDA kernel source for **Unweight** — lossless compression of BF16 MLP weights for LLM inference on NVIDIA Hopper GPUs (H100, H200).

This repository contains the encoding, decoding, transcoding, and reconstructive matmul kernels described in the technical report.

**Technical report:** https://research.cloudflare.com/nikulin2026

## Overview

BF16 exponent fields in trained LLM weights carry ~2.6 bits of Shannon entropy in their 8-bit allocation, while sign and mantissa fields are near-incompressible. Unweight separates each BF16 value into sign+mantissa and exponent bytes, Huffman-codes the exponents over a per-tensor 16-value palette, and handles rare exponents through verbatim rows rather than inline escape symbols.

The central inference primitive is a **reconstructive matrix multiplication** — a persistent [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) LCF kernel that reconstructs BF16 tiles in shared memory immediately before Hopper WGMMA consumption, eliminating a full HBM round-trip for the weight matrix.

Four execution pipelines — full decode + cuBLAS, exponent decode + reconstructive matmul, palette transcode + reconstructive matmul, and direct palette + reconstructive matmul — are selected per projection and batch-size bucket via coordinate-descent autotuning on end-to-end throughput. A hard/easy layer alternation schedule extends preprocess-compute overlap across layers with different encoding profiles.

On Llama 3.1 8B, Unweight achieves ~30% compression on MLP weights (~20% total model size reduction) with lossless numerical equivalence.

## Requirements

- NVIDIA Hopper GPU (SM 9.0a) — H100 or H200
- CUDA Toolkit 12.4+
- C++20 capable `nvcc`

## Building

```bash
git submodule update --init --recursive
make        # → build/libunweight.a
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
