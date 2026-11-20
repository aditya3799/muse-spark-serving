# Serving track: lossless SGLang optimization

This track applies a source patch to SGLang v0.5.23 to serve Muse Spark Open on **eight H200 GPUs**, with tensor/expert parallelism and DFlash speculative decoding. Our best completed scored run achieved **1.2363x stock throughput** at the judge's interactivity targets. The final submission uses the same source plus a cache-capacity flag that passed a separate 32-user check.

## Result and ranking status

Snapshot fetched on **20 November 2026 (UTC)** for `aditya3799`:

| Run | Scope | Result |
|---|---|---|
| `317c88a5fd474cd2` | Full scored ladder, Humming flag | **1.2363x**, correctness/later-turn checks passed |
| `2b583e18aad84d09` | Later joint projection candidate, full ladder | **1.2239x**, passed; lower score than the earlier release |
| `f464d5f09f534363` | Exact final source + cache ratio 0.35, unscored 32-user check | **1.398x indicative ratio**, passed; **-1.4 GB/GPU** versus stock |
| `9577d0fdbe714fa6` | Final scored submission, submitted 20 November at 12:23:30 IST | **Queued; final score pending** at fetch time |

**Serving rank: unavailable.** The [official serving board](https://openkernelchallenge.dev/api/v1/serving/leaderboard) still returned `hidden: true` and no rows. A score is not a rank, and the 1.398x one-load check is not the final four-load score. The earlier full-score change from 1.2363x to 1.2239x is about -1.00%; one comparison does not establish its statistical significance or cause.

[Saved receipts](results/) support these statements. The final receipt and reviewer notes are in [submission](submission/). The best full run's normalized ratios were 1.1706x, 1.1867x, 1.2963x, and 1.2974x at 1, 4, 16, and 32 concurrent users respectively.

## Submitted artifact

- Base commit: `297e54fd105e974cdf401e5b4d20253a22c16f90`.
- Patch: [`submission/patch.diff`](submission/patch.diff), **83,849 bytes** across **10 Python source files**.
- Patch SHA-256: `8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`.
- Final flags: `--moe-runner-backend humming --swa-full-tokens-ratio 0.35`.
- Longest added line: **226 characters**; published limits were 409,600 bytes and 400 characters per added line.
- Validation: plain `git apply --check` and plain `git apply` succeeded against the pinned base; all ten resulting source hashes match the frozen release and all ten Python files parsed.

[`source/python/sglang`](source/python/sglang/) provides the resulting modified files for browsing. It is an overlay, not a full SGLang checkout. The patch is the authoritative submission artifact. Its source is byte-identical to the earlier 1.2363x run; the final run changes the launch-time cache ratio.

## What changed and why

### Expert-parallel MoE execution

We integrated Humming expert execution with the existing NVFP4 representation, FP8 block scales, FP32 accumulation, and BF16 output boundaries. Canonical weights are reused, with local expert routing and scaling handled explicitly. Routing metadata can be reused without host reads of route values.

Device-side histograms partition light and heavy experts so sparse expert-parallel work does not always inherit a dense global tile choice. The larger-prefill extension preserves the original block-K and warp-K reduction geometry. Finite native configuration admission happens before capture; unsupported configurations fall back before an optimized launch. The cumulative patch also retains guarded adaptive Marlin alignment/tile support.

### Expert finalization

The finalization path separates independent expert RMS reductions from tiled weighted combination. This exposes more parallel work while preserving the per-slot combination's numerical boundaries. The source records the public MoE decomposition that informed the implementation.

### Projections and communication

For compatible replicated projections, larger prefills can partition token rows, while eligible small projections partition complete output columns. Gathering restores the original output order. Eligibility guards preserve the existing paths for unsupported geometries or communication workspaces.

### FA3 verification scheduling

The patch reuses scheduler metadata for compatible causal full-attention target verification instead of rebuilding the same schedule in each layer. Its guards cover query/KV geometry, causal mode, masks, and layer type. Sliding-window, local, cross-attention, and unsupported cases retain their existing scheduling paths. The submitted change does not replace the attention exponential or introduce lower-precision attention arithmetic.

### KV cache capacity

The final `--swa-full-tokens-ratio 0.35` flag uses SGLang's standard allocator to shift capacity toward the full-attention pool within the existing memory budget. It does not shorten the attention window, truncate prompts, or change request lengths. `--mem-fraction-static` remains 0.85.

The completed 32-user check reported 2,819,026 full-cache tokens and 986,659 sliding-window tokens. It passed correctness and later-turn checks with no short outputs, and reported first/later-turn TTFT ratios of 0.72/0.22 versus stock. This supports the final choice, but does not prove a gain across every load or dataset.

## Experiments that informed the final choice

We investigated alternate Humming schedules and packing, expert partitioning, projection sharding, DFlash projection/auxiliary-buffer handling, attention prologue fusion, communication changes, and cache pressure. These are recorded under [research](research/).

The later joint projection candidate passed the official full-model checks but scored 1.2239x, so it was not selected for the final source. Several local kernels showed isolated improvements without demonstrated serving gains. Unsupported-shape fallbacks, actual dispatch coverage, TP8 communication, GPU memory residency, and graph integration all limited what could be inferred from local measurements.

Local qualification used CPU contracts, changed-input CUDA graph tests, source/hash checks, and completed sanitizer runs for specific fixtures. Some local timings were affected by a low-power RTX 3050 state. Such measurements are kept as component evidence, not multiplied into a projected serving score. The exact final source has official full-model evidence; newly researched branches do not inherit that evidence automatically.

## Apply and run locally

Use Linux with the pinned runtime, eight suitable GPUs, access to both model checkpoints, and sufficient host/disk memory. The judge image was:

```text
lmsysorg/sglang@sha256:f832b5c39559c9caa900b22ad37c76fcce66470676855c42052102c537812f1f
```

From the repository root:

```bash
git clone https://github.com/sgl-project/sglang sglang-runtime
git -C sglang-runtime checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
git -C sglang-runtime apply --check ../serving-track/submission/patch.diff
git -C sglang-runtime apply ../serving-track/submission/patch.diff
pip install -e sglang-runtime/python --no-deps

export MODEL_PATH=/path/to/Muse-Spark-Open-816B-A42B-NVFP4
export DRAFT_MODEL_PATH=/path/to/Muse-Spark-Open-DFlash2
bash serving-track/scripts/launch_local.sh
```

The launch script binds to localhost and carries the pinned judge launch configuration plus the two final flags. It does not download weights or create compute instances. A single H200 can exercise components, but cannot reproduce this eight-GPU serving setup.

For a meaningful mock comparison, use separate stock and patched checkouts, the same checkpoints and requests, matched repetitions, and the same concurrency ladder. Record time-to-first-token, throughput, memory, output validity, and generated-token counts. Public or synthetic conversations are a mock workload; they cannot reproduce the private judge corpus or certify the official score.

## Refresh the pending result

These are read-only status queries:

```bash
okc serve status 9577d0fdbe714fa6 --json
okc serve leaderboard
```

Update the saved result and the README together after the official run finishes. There is no new submission required to reproduce the publication record. See [reviewer notes](submission/reviewer_notes.txt), [validation](submission/validation.json), and [attribution](../NOTICE.md).
"# muse-spark-serving" 
