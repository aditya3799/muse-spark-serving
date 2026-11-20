# DFlash target graph-entry staging

Deferring the first registry fill until the target graph owns its final inputs reduced local graph-entry time in both changed-input seeds on the RTX 3050. Both arms used the pinned registry, the frozen parent’s preplanned second copy and a CUDA-graph integer consumer. Six ordinary/padded batch sizes passed exact changed-input and padding checks. Each of the 12 adjacent rounds per case timed 20 replays per arm, alternating the candidate's placement relative to baseline and including a duplicate baseline. CUDA events and synchronized wall time agreed on direction. The timed region excludes the model and FA3 metadata. These microseconds carry no official H200 or whole-step gain estimate.

| Requests | Seed 7219 paired wall saving, µs | Seed 8842 paired wall saving, µs |
|---:|---:|---:|
| 1 | 85.1 | 62.1 |
| 4 | 43.5 | 58.5 |
| 7 padded to 8 | 25.8 | 53.2 |
| 16 | 58.3 | 63.2 |
| 31 padded to 32 | 41.6 | 65.5 |
| 32 | 56.9 | 60.9 |

The table reports median parent minus candidate in paired rounds. An independent analysis uses the geometric midpoint of both parent controls to calculate signed log-speed reward, plus per-case paired bootstrap intervals. Every cell’s 95% interval is above zero, with noisy magnitudes. [Signed controls](gpu_stage_signed.json) [Pilot measurements](gpu_stage_seed7219.json) [Independent seed](gpu_stage_seed8842.json) [Candidate and source provenance](manifest.json)
