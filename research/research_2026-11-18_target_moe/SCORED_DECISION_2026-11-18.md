# November 18 scored-run decision

## Scored outcome for job 50a2f4d696224d6b

Retire the isolated Hopper routed-expert candidate. The judge passed correctness and excluded every load. At 16 users, raw throughput was 6,450.1 tokens/s/GPU versus 15,546.7 for the previous 1.2363x scored build. First-turn p90 latency was 174.256 s versus 61.385 s on that build. The submitted implementation uses 43 expert launches per eligible layer for a 16,384-row prefill. The receipt provides no per-kernel timing to attribute the regression. [Scored result](native_package_02/scored_50a2f4d696224d6b.json)

## Pre-run assessment

Recommend the isolated Hopper routed-expert patch for today's opportunity to improve on the proven 1.2363x build. The patch is native_package_02/candidate.diff, SHA256 1ce48266f47574c5c0d4fd3c557b12f6d24811d03121f9bd9ea4e89965f6065c. Its sole launch override is --moe-runner-backend humming. A scored run requires the user's permission. Its unscored serving job is 0afdc0b191dd4b42.

## Paired numerical comparison

compare_full_math.py and paired_mma.cuh use the packaged CUDA route builder, packed decoder, activation gather, both projection epilogues and production finalizer. The same 32 full-size expert weight matrices per stage, activations, route IDs, route scores, BF16 scaling, and normalization supply the candidate and the scored Humming reference. The portable operand check uses SM86 warp matrix instructions, global loads and three ordered kernels in place of Hopper warpgroup matrix instructions, TMA and cooperative scheduling. The test covers complete 4096-wide reductions and raw routed outputs before and after finalization. It measures numerical agreement. It does not measure the Hopper pipeline's speed or synchronization.

Two independently generated weight banks with seeds 4318 and 7829 passed 17 fixtures and 18,046,976 finalized scalar comparisons in total. The second run counted active tokens separately. Across its nine fixtures, 98.84% to 99.82% of active-token BF16 outputs matched Humming bit for bit. The largest final relative L2 error across either run was 0.000453. A 1,024-token stress case sent all eight routes per token to the local bank. Its active-token bitwise agreement was 99.15% with relative L2 error 0.000307. Zeroing expert zero's W2 weights on the last input changed the final output by over 21% in relative L2 in both runs. The 385-, 753- and 1,024-row cases also exercised the chunk boundary. The official serving judge measures generated tokens and whole-model behavior separately.

Receipts are paired_math_01/results.json and paired_math_02/results.json. Each run directory also contains the captured source and process exit status. To reproduce the second run in the configured WSL environment, run launch_probe.py --script compare_full_math.py --seconds 120 --out NEW_DIR --weight-seed 7829.

## Performance evidence and limits

Five off-board or scored H200 kernel-track reports named submission_v139_wgs_kperm3 under ../muse_moe_w4a16/profiles relative to serving record 2.642-2.652x at 128 verify rows and 2.954-2.983x at 256 verify rows, versus the kernel track's SGLang baseline. Those are the donor implementation's isolated routed-expert ratios. The serving port changes the decoder, reduction schedule, BF16 rounding order and finalization boundary. The donor selects a different kernel for eight rows, so its one-user result offers little direct evidence for this port. Historical full-layer 16,384-row comparisons across different jobs yielded 1.136-1.268x for an earlier fused build, with uncontrolled changes. No exact-package matched H200 timing or whole-model peak-memory measurement is available.

The measured parent four-load frontier ratios were 1.1706, 1.1867, 1.2963 and 1.2974. A 2x finish requires another 1.618x over this parent. If the two higher-load ratios doubled and the two lower-load ratios stayed at the parent's values, the score would be approximately 1.748x. This illustrates sensitivity to all four loads and is not a forecast.

DFlash has a measured benefit for the isolated projection on the local RTX 3050, with eight-way all-gather and NCCL timing outstanding. Mixed chunking needs overlap between an admitted prefill and running decode requests to produce benefit; there is no measured serving uplift for the queued flag change. Combining either with the Hopper patch introduces an untested interaction. The proposed scored run isolates the expert change and tests all four loads. Submitting now would spend today's single scored slot, with throughput regression, token correctness, TTFT and memory still open.
