"""Score elapsed time for independent live decode requests and two frozen tree widths."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument("--cached-draft",action="store_true");args=parser.parse_args()
stem="wide_cached" if args.cached_draft else "wide"
doc=json.loads((here/(stem+"_live.json")).read_text())
rows=doc["requests"]
assert len(rows)==12 and len(set(r["name"] for r in rows))==12, "wait for all six held cases and six independent public requests"
arms=[doc["policies"][str(n)] for n in (12,16)]
if args.cached_draft:arms.extend([name+"_guard" for name in arms.copy()])
seeds={"held_":82391,"public_":89121,"all":824149}
results={}
for group,seed in seeds.items():
    group_rows=[r for r in rows if group=="all" or r["name"].startswith(group)]
    times=np.array([[r["arms"][k]["seconds"] for k in ["chain","control",*arms]] for r in group_rows])
    pairs=len(times)
    draws=np.random.default_rng(seed).integers(0,pairs,size=(12000,pairs))
    sums=times[draws].sum(1)
    obs=times.sum(0)
    ref=(obs[0]+obs[1])/2
    paired=(sums[:,0]+sums[:,1])/2
    ctrl=sums[:,0]/sums[:,1]
    ci_ctrl=np.quantile(ctrl,[.025,.975])
    results[group]=dict(requests=pairs,timed_tokens=sum(r["tokens"]-1 for r in group_rows),
        chain_seconds=float(obs[0]),control_seconds=float(obs[1]),parent_control_ci95=ci_ctrl.tolist(),arms={})
    for i,k in enumerate(arms,2):
        checked=sum(r["arms"][k]["checked"] for r in group_rows)
        serial_errors={a:sum(r["arms"][a]["wrong"] for r in group_rows) for a in ("chain","control",k)}
        pair_errors=sum(r["pair_token_differences"][k]+r["pair_token_differences"]["control"] for r in group_rows)
        for r in group_rows:assert all(r["arms"][a]["checked"]==r["tokens"] for a in ("chain","control",k))
        rel=ref/obs[i];tail=.025/len(arms);l,u=np.quantile(paired/sums[:,i],[tail,1-tail])
        # Two frozen widths. Require gain over the size of duplicate-parent drift.
        noise=max(ci_ctrl[1],1/ci_ctrl[0],1.)
        decision="INCORRECT" if any(r["pair_token_differences"][k] and not r["pair_token_differences"]["control"] for r in group_rows) else "PARENT_REFERENCE_MISMATCH" if pair_errors or any(serial_errors.values()) else "KEEP" if l>noise else "SLOWER" if u<1/noise else "UNRESOLVED"
        reward=-100. if decision=="INCORRECT" else 100*math.log(rel) if decision in ("KEEP","SLOWER") else None if decision=="PARENT_REFERENCE_MISMATCH" else 0.
        results[group]["arms"][k]=dict(tokens_checked=int(checked),serial_errors=serial_errors,pair_errors=pair_errors,
            seconds=float(obs[i]),speed_ratio=float(rel),request_adjusted_interval=[float(l),float(u)],confidence=float(1-2*tail),decision=decision,reward=reward,
            steps={a:sum(len(r["arms"][a]["steps"]) for r in group_rows) for a in ("chain","control",k)},
            extra_fallbacks=sum(r["arms"][k].get("fallback_verifications",0) for r in group_rows))
out=dict(protocol="Sum elapsed decode time at fixed per-request token counts, compared to arithmetic mean of the two chain times. Independent request bootstrap, two-sided Bonferroni interval across all tested policies in this run. Reject if any token disagrees with serial gold or the duplicate. The public writing-style request set is independent of selection. Qwen3-1.7B and matched DFlash2 at TP1, RTX 3050. Short Muse-style coding prompts from six public trajectories are model inputs only.",results=results)
out["draft_context"]=doc.get("draft_context","Re-evaluate the historical draft context projections in every forward")
(here/(stem+"_reward.json")).write_text(json.dumps(out,indent=2))
print(json.dumps(results))
