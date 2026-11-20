# Target dense projection qualification

The scored patch is 8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b. The H200 receipt identifies TP8/EP8 Humming with multicast. Humming sets the model's use_marlin feature flag. Layer construction grows the shared projection workspace to handle 512 × 8192 BF16 outputs when multicast is available. The stock default disables layernorm sequence parallelism, which requires another expert backend on Muse.

| Projection | Scored arithmetic and dispatch |
|---|---|
| Attention QKV/gate | Unquantized, bias-free column-parallel BF16 linear. The local output contains Q, K, V and optional O gate. The attention fast path can subsequently fuse QK norm, RoPE and KV store. |
| Attention O | Unquantized, bias-free row-parallel BF16 linear followed by the attention TP reduction. Neither attention projection uses the FFN's specific FP32-output row helper. |
| Router | Above eight tokens the Hopper router multiplies BF16 activations by the two packed BF16 parts of its FP32 router weight, generates FP32 partials and adds hi/lo in FP32. At eight or fewer, the SM100-only CuTe router is ineligible on H200 and this path uses the ordinary FP32 linear. |
| Pool / unpool | H=8192 and expert H=4096. At M256, compute 512/1024 columns respectively with FP32 destination then cast to BF16, retaining full K. At M753, both operate on all rows and columns through bf16_gemm_dispatch's ordinary BF16 F.linear fallback. At M16384, each rank computes 2048 complete rows with FP32 destination then casts to BF16 before gathering. The older unpool multicast route wins first at M≤8 and uses the BF16-output dispatcher. |

Verified sources are the scored muse_spark.py, muse_spark_fp4_moe.py and muse_spark_gemm_ag.py, plus the pinned unquant.py and muse_spark_all_gather.py. Actual per-op traces were unavailable in the scored service receipt. Candidate kernels were discovered via vendor heuristics and checked with the installed cuBLASLt. The benchmark includes every product in the selected rank-local operator and its original output cast. No peer GPU work or transport is timed.

The first measurement under paired/ failed the changed-input graph check because its cuBLAS handle was invoked with the eager stream. Its captured outputs were stale. Discard the complete directory for timing analysis. The corrected run under paired_stream/ obtains the invoking CUDA stream at every call and passed every fresh result, control and intermediate check. Two independent seeds varied both inputs and weights over random, cancellation and finite mixed-magnitude designs. The search used separate calibration data; each FP32-destination survivor matched the complete unrounded FP32 matrix bitwise.

| Projection and host rows | Pilot native → candidate median µs | Holdout native → candidate median µs |
|---|---:|---:|
| Pool 256 | 204.8 → 208.2 | 166.4 → 165.4 |
| Unpool 256 | 219.5 → 205.6 | 223.6 → 205.2 |
| Pool 753 | 3332.3 → 3364.2 | 3280.9 → 3280.9 |
| Unpool 753 | 3325.9 → 3324.9 | 3288.0 → 3290.1 |
| Pool 16384 | 9950.2 → 9190.8 | 9685.4 → 9139.6 |
| Unpool 16384 | 10907.6 → 9491.3 | 10884.0 → 9565.2 |

Equal-weight plugin estimates for the six local operators were 1.0453× (1.0358–1.0547) and 1.0500× (1.0356–1.0650). Pilot classified MIXED; holdout classified NOISY and its small unpool duplicate control drifted. The two large operations were individually favorable in each seed. Aggregate plugin eligibility was false. These are single-RTX-3050 measurements; actual H200 kernel selection and service latency remain unmeasured.

A production implementation would warm representative dynamically observed tensor layouts after weight load and before graph capture, confirm all comparison pairs at the actual output dtype, benchmark against native with a duplicate control, then register only shapes with a reproducible net improvement. Candidate descriptors must be scoped to the live device, driver, cuBLAS version, math mode, input and output dtype, matrix strides, alignment and workspace capacity. Allocate per-stream scratch before capture; read the current CUDA stream at every launch. Use the scored local projection for absent or unsuitable plans and route both branches to the same original collective. The research search enumerated 56–250 checked configurations per shape. Actual cold discovery + screening cost must be measured at startup; storing fitted challenge shape/algo IDs in the serving patch would defeat portability. Exact sample outputs do not prove equality on untested H200 kernels or all possible inputs.


## Unchunked 16,384-row scored Humming FFN

Every measured arm computes the original rank-local pooler output for 2,048 rows, pre-expert weighted RMSNorm, complete sparse W13–SiLU–W2 at all 16,384 rows, scored post-expert norm and route weighting, and all 8,192 unpooler columns for this rank's 2,048 returned rows. Other ranks' pooled rows are precomputed outside the measurement. The EP reduction is represented by the local partial; attention, router execution, residual and communication are outside scope. The harness retains one live eager arm at a time on the 6-GiB RTX 3050.

The experiment re-enumerated runtime vendor configurations and recovered the two configurations selected by the earlier isolated trial on the same device. It alternated four timing orders between each sequence and its reverse, with four repetitions per distribution per seed. Baseline and control call the identical original path. Dense changes the two local GEMMs only. Joint additionally replaces the large-route scaling and SiLU boundary with the previously qualified native group-8 activation. Every changed-input trial checked the actual pool and unpool FP32 intermediates and complete BF16 FFN outputs, including signed-bit comparisons, finite outputs and distinct input/output fingerprints. All checks passed.

| Seed and routes | Parent ms | Dense ms | Dense + group-8 ms | Native copy ms |
|---|---:|---:|---:|---:|
| Pilot, uniform | 164.63 | 159.54 | 150.75 | 155.74 |
| Pilot, clustered | 653.14 | 645.86 | 615.42 | 640.17 |
| Holdout, uniform | 170.79 | 164.01 | 160.13 | 169.23 |
| Holdout, clustered | 657.00 | 658.02 | 637.23 | 657.89 |

The independent holdout plugin estimates were 1.0160× (1.0094–1.0227) for dense and 1.0501× (1.0345–1.0659) for joint. Pilot estimates were 1.0193× and 1.0618×; its identical control failed the noise gate. Neither mechanism passed the two-run promotion criterion. The held-out parent/control paired medians were 1.0027× uniform and 0.9992× clustered. A separate group-8-only arm was not included in this fixture.

Peak measured live tensor allocation was 4.480 GiB per arm group. Both selected descriptors use zero vendor scratch. Each of the two layouts returned 56 checked configurations. Complete FP32 qualification retained 23 pooler and 20 unpooler choices. Local cold discovery, full-matrix stress screens and timing for the pair took 7.44 seconds on the first process and 4.05 seconds on the second. Each unique device/layout can share a plan across layers after application-specific validation. The search's 32-MiB-cap scratch is separate from the selected plans' zero-byte requirement. H200 startup, numerical ordering, capture replay and end-to-end behavior remain unmeasured.

Raw outputs are in [full pilot](full_pilot/detail.json), [full independent holdout](full_holdout/detail.json), [dense scoring](dense_whole_reward.json), and [joint scoring](joint_whole_reward.json).


## Reproducing the pinned serving candidate

[candidate.diff](candidate.diff) applies to SGLang 297e54fd105e974cdf401e5b4d20253a22c16f90. Its SHA256 is faa6f54f10479b016062c0f1afed36a904de84c0efba5560cf95c45d83982c1e. [delta.diff](delta.diff) applies to the scored 8c754721…c2b Humming patch. Running `python package.py` materializes only the required files in a temporary Git repository, applies both forms with whitespace checking and verifies identical final file contents.

The patch adds the previously qualified group-8 activation, then a method called before each eligible target prefill graph capture. The model enumerates the unique shapes of Humming's actual pool and unpool weights and warms complete-row full-K BF16×BF16→FP32 products. The candidate tries runtime vendor configurations with zero execution scratch, checks complete FP32 outputs on the actual weight and finite synthetic/cancellation weights, then compares against native and a duplicate control. A shape is enabled only on positive balanced timings; declining a layout preserves torch.mm. The existing cast to BF16 and the common all-gather remain after the selected GEMM. Caches are local to the process/device, require the same math flags and aligned contiguous BF16 tensors, and carry no challenge constants or cached outputs. Registered plans use the live CUDA stream. Previously unseen/capture-only layouts, missing compiler/headers and loader errors fall back. The search can use up to 32 MiB as the search cap; selected GEMMs require zero scratch.

On the local Ubuntu CUDA 13 environment, `CUDA_VISIBLE_DEVICES=-1 python test_package.py` compiled the shipped C++ implementation through the production Torch extension loader with the benchmark virtualenv on PATH. The [capture smoke](hook_smoke.json) used the patched runner method, model hook, communication helper and registered production op with a real nondefault CUDA stream. The real device gate rejects SM86. The test overrode only the Hopper capability predicate in the model hook, and represented gather as an identity because other ranks were unavailable. The selector retained the native pooler and enabled the unpooler. Five changed-input replays for both outputs and all three native fallbacks (dtype, unknown shape and misalignment) matched the original bits. H200, genuine TP8 transport, and the complete initialized server are untested.

The single-layer D:/OpenKernelChallenge/muse_prefill_layer donor references SGLang a55ab7d5. Its baseline uses FP8 paged FA3, identity TP reductions, an older replicated pooler and Marlin experts. The scored startup log explicitly selects FA3. SM86 cannot execute the H200 FP8 FA3 path. The donor reference's short cases materialize dequantized KV and use PyTorch softmax. At M128 the packaged changes would both bypass; at M753 only the group-8 boundary executes. No surrogate full-decoder timing was reported.

The initial full-FFN benchmark measured frozen locally selected descriptors directly. The later prospective full-FFN trials call the actual packaged dynamic registration and Torch custom op. Their declared seeds were 260519 and 721903, with eight alternating/reversed timing orders per route distribution and two extra complete warmup rotations. The original pilot, including its noisy duplicate control, remains part of the evidence.

The SM90 predicate is inherited from the scored parent, not introduced by this optimizer. Its `prefill_row_projection_supported` explicitly requires compute capability `(9, 0)` at line 87 in the scored `muse_spark_gemm_ag.py`. A TP8 SM86 installation therefore takes a different parent path. Only the model's capability predicate was overridden in the local smoke to reach the same rank-local arithmetic and test the real NVIDIA primitive; collective transport remained an identity.

## Prospective production-selector result

Fresh process/weights per seed. All 96 native-copy, candidate and joint end-to-end comparisons were bitwise equal, and the FP32 intermediate checks passed on both legs. The duplicate-control point differences were 0.13% (95% interval −0.23% to +0.47%) and 0.25% (−0.77% to +1.83%). Peak live tensor allocation was 4.480 GiB. The first real selector registered unpool only in 8.12 s; the second registered both layouts in 4.06 s. The execution plans used zero workspace.

| Runtime-qualified arm | Seed 260519 speedup and 95% interval | Seed 721903 speedup and 95% interval | Paired plugin outcome |
|---|---:|---:|---|
| Dense only | 0.9929× (0.9832–1.0017) | 1.0043× (0.9936–1.0172) | NO-GAIN / NO-GAIN |
| Dense + group-8 | 1.0281× (1.0246–1.0312) | 1.0408× (1.0284–1.0570) | KEEP / KEEP |

The combination improved this complete unchunked rank-local FFN on both seeds. These data do not establish a separate contribution from dense tuning over group-8 alone. Compare the [first raw trial](prospective_a/detail.json), [second raw trial](prospective_b/detail.json), [joint reward](prospective_joint_reward.json) and [dense-only reward](prospective_dense_reward.json). The two live-GPU runs used `python full_ffn.py --production --seed 260519 --pairs 8 --out prospective_a` and `python full_ffn.py --production --seed 721903 --pairs 8 --out prospective_b` with the pinned CUDA 13 Linux environment. The 16K expert dispatch is unchunked and eager. A true H200 end-to-end comparison and numerical validation on service inputs are still absent. [FA3 hardware requirements](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)

