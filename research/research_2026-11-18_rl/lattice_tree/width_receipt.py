"""Record paired duration and simulated commit-count evidence without treating simulated tokens as verified outputs."""
import json
import statistics
from pathlib import Path

here=Path(__file__).resolve().parent
reports={}
for name in ("pilot","holdout"):
    raw=json.loads((here/("gpu_widths_"+name+".json")).read_bytes())
    rows={}
    for shape,result in raw["shapes"].items():
        samples=result["rows"]
        base=sum(x["timing_us"]["baseline"] for x in samples)
        candidate=sum(x["timing_us"]["candidate"] for x in samples)
        commits_a=sum(x["ref_commits_proxy"] for x in samples)
        commits_b=sum(x["candidate_commits_proxy"] for x in samples)
        rows[shape]=dict(samples=len(samples),gpu_step_ratio=base/candidate,
                         counterfactual_commits_ratio=commits_b/commits_a,
                         counterfactual_tokens_per_second_ratio=base*commits_b/(candidate*commits_a),
                         median_paired_step_ratio=statistics.median(x["timing_us"]["baseline"]/x["timing_us"]["candidate"] for x in samples),
                         different_bf16=sum(x["bit_mismatches"] for x in samples),
                         largest_relative_l2=max(x["relative_l2"] for x in samples))
    reports[name]=rows
penalty=-100 if any(row["different_bf16"] for r in reports.values() for row in r.values()) else None
data=dict(method="GPU step times include row selection and the original complete rank-local Humming expert pair plus finalization. Committed token counts are resampled from observed public GLM per-request rates, not actual Muse outputs.",
          baseline_sha256=raw["baseline"],decision="INCORRECT" if penalty else "UNCHECKED",reward=penalty,runs=reports)
(here/"gpu_width_summary.json").write_text(json.dumps(data,indent=2))
print(data["decision"],[(k,round(v["counterfactual_tokens_per_second_ratio"],3)) for k,v in reports["holdout"].items()])
if penalty is not None:
    import hashlib
    import subprocess
    import sys
    failed=dict(baseline_sha256=raw["baseline"],candidate_sha256=hashlib.sha256((here/"width_gpu.py").read_bytes()).hexdigest(),
                agent="decode-width",candidate="early adaptive DFlash verify compaction",scope="Rank-local Humming with graph-rounded widths; live BF16 outputs compared to the frozen width-eight source",seed=raw["seed"],
                active=True,complete=True,correct=False,evidence="gpu_widths_holdout.json",summary="gpu_width_summary.json")
    receipt=here/"width_ablation.json"
    receipt.write_text(json.dumps(failed,indent=2))
    subprocess.run([sys.executable,str(here.parent/"score_trials.py"),str(receipt),"--output",str(here/"width_reward.json")],check=True)