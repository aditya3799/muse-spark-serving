# Native BF16 scales for indexed Humming

Widening immutable W2 group scales slowed the complete local FFN to **0.98243×** the frozen scored parent, with a paired interval of 0.98160–0.98328 and signed reward −1.773. Every changed-input finalized BF16 output matched, as did raw native W2 outputs on ordinary and mixed-exponent inputs. The native scale representation used all 254 finite signed E4M3 encodings per expert in both seeded W13 and W2 weight banks. The discovery failed performance qualification, so no holdout or serving run followed.

| Local input rows | Normal, parent / candidate median | Skewed, parent / candidate median |
|---|---:|---:|
| 128 | 0.9684× | 0.9747× |
| 768 | 0.9859× | 0.9964× |

Each of four cells had eight reciprocal-order, changed-input CUDA graph rounds and duplicate original controls. Metrics cover the local TP-shaped dense projections, normalization, complete 32-expert W13/activation/W2/post-norm rank, with external rank inputs staged beforehand. The GPU was a power-limited RTX 3050 Laptop. H200 end-to-end and on-device speed were not measured.

Converting scales at weight preparation leaves each FP4 nibble intact and supplies the mathematically same native operand. All 4,064 finite nibble/scale pairs match in the CPU bit audit. The original effective exponent correction is 246−124−120=2. Native BF16 scales use 126−124=2. Consequently both modes have the same fourfold epilogue restoration. Representative fully compiled SM90a W13 and W2 kernels each retain 32 HGMMA sites while removing 34 HMUL2, 34 PRMT and 68 LOP3 static sites. W13 goes from 85 to 87 registers and W2 stays at 87. Each adds 8 KiB shared memory in these two geometries. The full M128/M768 light and heavy matrix screen finds 0–8 KiB of added shared memory and unchanged or slightly reduced stack. These binary counts did not predict a gain on the local card.

For 32 experts of width 4096, permanent net widening adds 32 MiB per layer for W2 and 64 MiB for W13. The amount assumes replacement of each canonical scale tensor. The frozen loader already removes raw native aliases after packing. The local paired harness retains the original reference intentionally. Over 60 routed layers W2 alone would add 1.875 GiB per rank and both stages would add 5.625 GiB. Conversion temporarily retains the old tensor while allocating the new one. No checkpoint/KV memory increase was exercised or credited.

[Arithmetic and memory proof](proof.json) · [Hopper binaries](binary_counts.json) · [Full Hopper geometry screen](hopper_scale_geometries.json) · [Raw native and paired outputs](pilot.json) · [Canonical FFN samples](pilot_w2.json) · [Signed reward](pilot_w2_reward.json)
