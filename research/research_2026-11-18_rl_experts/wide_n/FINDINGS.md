# Full-K CTA width 512

The W2-only candidate scored 1.0073× in discovery [1.0003, 1.0142] and 1.0036× in the independent holdout [0.9962, 1.0115]. The plugin classified the holdout NO-GAIN. W13-only scored 1.0044× [0.9979, 1.0122] in discovery, and changing both stages scored 1.0006× [0.9927, 1.0086]. Both were NO-GAIN. No configuration earned two-seed promotion.

The sole changed parameter was the width of an independently computed output-column CTA tile, 256→512. The entire warp shape, block and warp K, accumulator type, route alignment, stage count and launch grid remained fixed. The admissibility table contains measured binary resource usage and static memory limits. Included kernels have no local spills, fit the device register limit and fit block shared memory. The two previously frozen M128 shapes require 123,904 and 148,480 dynamic bytes on SM86 at three stages. The GPU allows 99 KiB per block. Those M128 partitions were excluded before timing. On the actual RTX runs the admitted three-stage local geometries were [16,512,64]/[16,64,32] and [32,512,64]/[16,64,64]. All admitted warps retained the scored reduction. The full device-hidden screen admitted eight of 20 SM86 stage/configurations and six of ten SM90a stage/configurations. The SM90a set is compile evidence only. The first disassembly attempt used an unavailable WSL executable; [resources_checked.json](resources_checked.json) contains the completed Windows CUDA disassembly and supersedes that preliminary output.

| Discovered W2 padded work that used width 512 | Normal | Skewed |
|---|---:|---:|
| 8 rows | 100% | 100% |
| 32 rows | 93.5% | 14.8% |
| 768 rows | 26.0% | 0.4% |

These shares count padded Humming rows, separately for W2, using live device counters on each graph replay. At M768 skewed the wider W2 had work in two of eight discovery iterations. The original configuration handled all other partitions. Trials used the official kernel-task normal/skewed EP8 busiest-rank generator, eight paired rounds per shape, duplicate parent controls and finalized bitwise comparisons. The holdout used a distinct weight/route seed and reversed the row order. Direct stage checks also compared full raw outputs, inactive sentinels, and mixed-exponent input patterns. The canonical result includes the six shape/distribution pairs that had some nonzero widened work on each seed. [Raw discovery](pilot.json) · [Raw holdout](holdout.json) · [Paired verdict and intervals](holdout_w2_reward.json)

