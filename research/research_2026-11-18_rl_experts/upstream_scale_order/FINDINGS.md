# Upstream Humming scale register order

The frozen Humming helper gained 0.63% in the first RTX 3050 GPU trial and 0.70% in the independent holdout. The gain was largest at eight verification rows. Both trials earned KEEP from the shared paired estimator. The intervals were 1.00482–1.00771× and 1.00489–1.00938×, respectively, across all eight equally weighted cases. These are local routed-MoE timings. Runtime H200 and serving scores remain unmeasured.

| Verification rows | Holdout uniform, parent → candidate (µs) | Holdout clustered, parent → candidate (µs) |
|---|---:|---:|
| 8 | 1,590 → 1,559 | 2,744 → 2,696 |
| 32 | 4,500 → 4,499 | 4,850 → 4,848 |
| 128 | 7,400 → 7,401 | 7,908 → 7,849 |
| 256 | 7,623 → 7,586 | 10,779 → 10,651 |

The candidate reads pairs of E4M3 group scales in the order produced by the pinned register decoder. It removes the explicit swaps at offsets one and two in each four-scale group. The register selection supplies the same raw bits to each multiply. All floating-point operations, accumulator partitions and the epilogue retain their previous inputs. Complete compiled H200 W13 changes from 27 to one PRMT site, and W2 changes from 34 to zero. Both preserve the count of HGMMA sites. The source attribution and Apache 2.0 license are in this directory. [Humming PR 107](https://github.com/vllm-project/humming/pull/107)

Both native randomized trials used the released Humming parent (patch SHA256 `8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b`), CUDA graphs, full W13 → scored BF16 scaling and SiLU → W2 → scored RMS and weighted sum, 32 local experts, width 4096, top eight, pair reversal, duplicate original controls and fresh input/weight seeds. Eight measurements per arm per case were collected. The fixture has a second pattern with eight hot local experts on every fourth token. Each expert received all 254 finite E4M3 scale encodings on W13 and W2. Separate stage measurements used uniform routing (one populated partition at each shape). All active W13/W2 results and both full-route-pattern finalized results passed bitwise BF16 checks. The CPU register selection also covers both reserved E4M3 encodings.

The alternate exponent-folding change fails the frozen helper's output contract. With one nonzero weight `+1`, group scale `+1` and a finite normal BF16 activation `0x0301`, the original computes `BF16(A/4) × 4`, producing `0x0301`. The public fold computes `BF16(A/128) × 128`, producing `0x0300` under gradual underflow. Flushing subnormal intermediates also changes the answer. The exhaustive unit-weight check found 1,768 mismatches across 65,280 finite BF16 inputs under the no-flush model. The cause is the two different scale factors before the parent converts its FP32 accumulation to BF16. This branch was retired before GPU testing. [Humming PR 103](https://github.com/vllm-project/humming/pull/103)

The L2 raster suggestion overlaps the prior `swizzle_e8_m3012_01/results.json` whole-Humming comparison. Ratios at 3,012 rows were 0.994× for uniform, and 1.000× for heavy-local-skew, in both warm and cold tests. Switching tiny-M Marlin configs overlaps the local EP-aware plan. Upstream's weight-eviction hint on A100 provides 5–9% kernel latency reduction at M8 on block-FP8; the present FP4 parent uses non-TMA weight copies for the archived W13/W2 H200 kernels, where upstream disables the hinted instruction on compilers earlier than CUDA 13.4. None of these donate a measured competitive gain to the current score. [Humming eviction experiment](https://github.com/vllm-project/humming/pull/152)

The canonical paired receipts are [pilot.json](pilot.json), [holdout.json](holdout.json) and [reward.json](reward.json). [cpu.py](cpu.py) regenerates the exhaustive arithmetic proof. [offline.json](offline.json) and [sass_counts.json](sass_counts.json) record the H200 compile evidence. The cumulative local candidate is [candidate.diff](candidate.diff), SHA256 `9ea3824c56c110645778bc54d3bea34cd8622a8c7d554f978c01e6a7502acbca`.

