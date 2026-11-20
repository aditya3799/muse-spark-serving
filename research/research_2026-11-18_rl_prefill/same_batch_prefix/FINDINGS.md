# Shared-prefix admission during cold prefill

Hold this patch for serving. Across all four loads in the judge-length stress, it saved zero prefill rows on all three public corpora. The official H200 result reports 0.92 aggregate cache hit at each load, equal to the judge's expected hit rate. The official conversations and per-request cache hits are hidden. [Scored receipt](../../release_2026-11-18/scored_submission_status.json) [Judge description](https://openkernelchallenge.dev/)

The isolated patch compares a waiting request's unmatched tokens with an already-admitted request in the same cache namespace and on the same radix node. When the admitted request will compute at least half of the work that fits in the current pass, the other request waits for publication. The next admission uses the ordinary radix match and obtains the published cache indices. The default FCFS scheduler does not otherwise compare these requests. SGLang's longest-prefix-match policy has a related advisory check against a simulated waiting-queue tree. [Patch](delta.diff) [Pinned scheduler policy](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_policy.py) [Radix cache design](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/)

## Admitted prefill rows at 16 users

| Inputs | Requests | Median first-prompt tokens | Rows saved with immediate publication | Rows saved with one-pass publication delay |
| --- | ---: | ---: | ---: | ---: |
| Public Muse trajectory fields | 20 | 1,004.5 | 44.30% | 39.73% |
| OpenHands messages | 491 | 1,720 | 25.51% | 37.34% |
| SWE-smith messages | 40 | 578 | 0% | 0% |
| Public first prompts with private tails extended to 74,160 | First 32 or fewer per corpus | 74,160 | 0% | 0% |
| Positive control of identical 74,160-token prompts | 32 | 74,160 | 10.41% | 0% |
| Later turns on distinct resident branches, 753 new tokens per request | 32 | — | 0% | 0% |

The 16,384-token prefill chunk fills a batch before another long cold request can enter. Its shared system prefix has usually been published by the time a second request fits. The public short prompts fit concurrently. They also expose a batching cost. At four users the original and changed scheduler needed 5 and 10 batches on the 20 Muse prompts and 123 and 244 batches on OpenHands. A row reduction at these lengths cannot establish a wall-time gain. The private-tail fixture preserves public prefix overlap and sets length to the official mean. Different hidden overlap or arrival times can produce a different outcome. [Raw replay](replay.json) [Pinned Muse dataset](https://huggingface.co/datasets/PatronusAI/few-word-do-trick-traces/tree/8cfdcf2d8384d0d765eca390e5bfba067e3b2899)

The scorer's original patch SHA-256 is `8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`. The builder verifies it and checks both files against pinned commit `297e54fd`. The isolated delta compiles and passes `git apply --check`. The replay directly executes the candidate method, compares 600 generated requests against a deterministic causal token-state reference, and tests namespaces, page rounding, protected inputs and altered prompt IDs. The cold and warm workload trials ran on CPU. The evidence supplies no H200 speed estimate. [Builder](build.py) [Replay](replay.py)

