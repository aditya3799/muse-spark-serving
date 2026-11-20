# Controlled CPU poll results

The kernel-autoresearch plugin groups neighboring rounds with reversed arm order, averages paired log times and bootstraps the five adjacent AB/BA pairs per seed. The signed stage reward is 100 × ln(parent time / candidate time). A duplicate parent supplies the noise guard. The eight ranks measure one local placement of the scheduler control stage. They do not constitute eight independent placements. [Plugin scoring program](score_cpu.py) [Per-case intervals and noise decisions](cpu_signed_rewards.json)

| Local receive pattern | Pilot signed reward and 95% ratio interval | Holdout signed reward and 95% ratio interval | Noise-controlled conclusion |
| --- | --- | --- | --- |
| Empty | +33.40 [1.386, 1.409] | +14.65 [0.991, 1.272] | Duplicate-parent difference resolved away from one in each run |
| One arrival in eight | +25.22 [1.247, 1.329] | +34.13 [1.292, 1.568] | Both held above the duplicate noise threshold |
| All arrivals | +2.50 [0.999, 1.052] | −1.06 [0.959, 1.027] | Effect unresolved |
| Equal-weight diagnostic | +20.38 [1.209, 1.243] | +15.91 [1.104, 1.237] | Both held under assumed equal distribution weights |

The H200 arrival mix, absolute Gloo latency, HiCache queue status and exposed CPU scheduling share are unknown. The equal weighting is a local diagnostic. The retained launch explicitly prints scheduler_recv_interval=1. The pinned default enables the ring, but the official log does not print its constructed handle. [Pilot canonical](canonical_cpu_7951.json) [Holdout canonical](canonical_cpu_18272.json)

A separate eight-rank run created two disjoint four-rank process groups. Their global source ranks were 0 and 4, as determined by the factory's local writer index zero. Across 12 cases per rank and arm, exact post-receive wrapper, MM, and shared-feature hook order survived even when an empty poll had pending MM work. The absent-ring and simulated remote-handle paths used real Gloo. Source-local aborts rode with their own subgroup's payload, and non-leader local lists were ignored by the original and candidate, as expected. [Subgroup results](subgroups.json) [Executable check](subgroups.py)
