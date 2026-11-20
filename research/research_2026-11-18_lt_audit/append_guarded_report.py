import json
from pathlib import Path
here=Path(".")
data=json.loads((here/"guarded_aa.json").read_text())
def fmt(value):
    return f"{value['gain']:.4f} [{value['ci'][0]:.4f}, {value['ci'][1]:.4f}]"
text=["","## Frozen guarded production result","",
"The 110,991-byte cumulative patch passed a clean staged application against SGLang 297e54fd105e974cdf401e5b4d20253a22c16f90. Its SHA-256 is \`84eff9bc0fd443eba15e3b2f23ca8985e7776c562f6d5746255613df5924ce18\`. The scored parent SHA-256 is \`8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b\`. Every changed path is permitted source under python/sglang. The longest added line has 226 characters. The patch contains no compiled binary, rename or model-loader change.",
"",
"On the real checkpoint the scheduler-stream startup accumulated 13.487 seconds, including local first-use compilation and discovery. It independently admitted 128 and 256 rows. The 8- and 32-row projections took the native branch. All sixteen changing-input test cells matched at the FC, hidden normalization, dense KV and physical KV levels. Each seed checked 347,494,400 cached values per arm. This exercised the exact patched worker prewarm and model methods with fixture implementations for unrelated services and upstream target layers. It ran on one physical TP-rank-zero slice and did not launch the full SGLang serving stack.",
"",
"The plugin's aggregate pilot verdict was **KEEP**, 1.05868× [1.04942, 1.06729]. The independent held-out verdict was **MIXED**, 1.06609× [1.06059, 1.07172]. Local promotion was false. [Formal paired reward](guarded_reward.json)",
"",
"### Native-bypass controls","","Ratios are paired plugin speedups. Brackets give 95% bootstrap intervals. Each arm was independently executed with the same changing inputs and reverse-order pair schedules.","",
"| Seed | Batch and commit | Candidate / native | Native control / native | Candidate / native control |",
"|---|---|---:|---:|---:|"]
for stage in ("pilot","holdout"):
    for name,r in data[stage]["shapes"].items():
        if name.startswith(("B1_","B4_")):
            text.append("| "+" | ".join([stage,name,fmt(r["candidate_vs_baseline"]),fmt(r["control_vs_baseline"]),fmt(r["candidate_vs_control"])])+" |")
text+=["","Held-out B4 partial triggered MIXED. A second execution of the unchanged native control also slowed relative to baseline, while the candidate-versus-control interval included parity. That control can account for the observed B4 partial difference within measurement uncertainty. It cannot rule out small native-branch overhead. Both held-out B1 candidate-versus-control intervals were slightly below parity, although neither candidate-versus-baseline interval excluded parity. The secondary control comparisons do not supersede the formal verdict. [Control comparisons](guarded_aa.json)","","### Active real-weight production timings","","The held-out values below are synchronized median wall milliseconds for the complete measured context path; the interval uses the paired plugin statistic.","","| Batch and commit | Native ms | Candidate ms | Native control ms | Candidate / native |","|---|---:|---:|---:|---:|"]
for name,r in data["holdout"]["shapes"].items():
    if name.startswith(("B16_","B32_")):
        wall=r["median_wall_ms"]
        text.append(f"| {name} | {wall['baseline']:.3f} | {wall['candidate']:.3f} | {wall['control']:.3f} | {fmt(r['candidate_vs_baseline'])} |")
text+=["","Earlier packaged-primitive CUDA graph checks matched all 20 fresh-input replays on nondefault and alternate streams. The full checkpoint SHA-256 for fc.weight was \`036d2e2e85e16a4755f00b234fbe932095949a0d44a4ff7b2e4df821ba959b0a\`. The combined verified auxiliary blob SHA-256 was \`0aa5203bbc30522ff878f55f7427974004ac42e24c1ae04cd3af2d7e7e4c0309\`. Eight-H200 warmup cost, actual complete-model token agreement, TP-wide algorithm consensus, the scheduler's maximum peak memory and serving score were not measured. [Patch](integrated_candidate.diff) · [Held-out details](guarded_holdout_detail.json) · [Source-package graph check](anchor_report.json)"]
with (here/"INTEGRATION.md").open("a",encoding="utf8") as f:f.write("\n".join(text).replace("\\\`","\`")+"\n")

