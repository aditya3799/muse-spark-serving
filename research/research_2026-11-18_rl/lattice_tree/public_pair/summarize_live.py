"""Paired request-level throughput ratio and correctness for the real, local verifier."""
import json
import math
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent
doc=json.loads((here/"live_tree.json").read_text())
rows=doc["comparisons"]
assert len(rows)==6 and sorted(r["id"] for r in rows)==[1,3,5,7,9,11]
times=[];steps=[];wrong=[];n=0;eq=0
for row in rows:
    parent=row["arms"]["chain"]
    tree=next(v for k,v in row["arms"].items() if k!="chain")
    assert parent["checked"]==tree["checked"]
    n+=parent["checked"]-1
    eq+=len(parent["produced"])
    times.append([parent["seconds"],tree["seconds"]]);steps.append([len(parent["steps"]),len(tree["steps"])])
    wrong.append([parent["wrong"],tree["wrong"],row["pair_token_differences"]])
times=np.array(times);steps=np.array(steps);t=times.sum(0)
ratio=t[0]/t[1]
rng=np.random.default_rng(75419)
boot=np.array([times[rng.integers(len(rows),size=len(rows))].sum(0) for _ in range(4000)])
ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975])
diff=np.array(wrong).sum(0).tolist()
verdict="INCORRECT" if any(diff) else "SLOWER" if ci[1]<1 else "FASTER" if ci[0]>1 else "UNRESOLVED"
reward=-100. if any(diff) else 100*math.log(ratio) if verdict!="UNRESOLVED" else 0.
out=dict(scope="One RTX 3050, real public Qwen3 1.7B and DFlash2, one active request. Compiled scored chain selector and compiled tree selector, shared eager SGLang transition expression. Both arms include draft and target forwards, decision and real KV/tap commit. Prompt prefill excluded. Six request pairs; no duplicate-parent timing arm; variation across trials is not independently replicated. The ratio is not Muse/H200 service throughput.",
         paired_token_differences=diff[2],serial_target_differences_baseline=diff[0],serial_target_differences_tree=diff[1],
         tokens_per_arm_checked=eq,timed_generated_per_arm=n,totals_s=t.tolist(),
         steps_per_arm=steps.sum(0).tolist(),local_tokens_per_s=(n/t).tolist(),paired_speed_ratio=float(ratio),
         paired_request_interval=ci.tolist(),per_request=(times[:,0]/times[:,1]).tolist(),
         signed_log_effect=100*math.log(ratio),decision=verdict,reward=reward,planner=doc["planner"],
         fixed_shape_future_check=doc["masked_future_check"])
(here/"live_reward.json").write_text(json.dumps(out,indent=2))
print(out)
