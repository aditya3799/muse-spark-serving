"""Invert the historical frontier proxy to express the acceptance and prefill requirements for a target score."""
import json
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root/"research_2026-11-18_frontier"))
from frontier_backtest import projected_curve, vectors

def main():
    data=json.loads((root/"release_2026-11-18/scored_submission_status.json").read_bytes())
    result=data.get("response",data)["result"]
    rows=result["rungs"]; shape=result["shape"]; turns=shape["turns"]
    points=result["frontier"]["points"]
    x=[p["tok_s_per_user"] for p in points]
    stock=[p["stock_tok_s_per_gpu"] for p in points]
    reconstruct=projected_curve(*vectors(rows,[1.]*len(rows)),x,stock)
    observed=result["frontier"]["score"]
    phases=[(r["ttft_first_p90_s"]+(turns-1)*r["ttft_later_p90_s"],
             turns*shape["output_mean"]/r["tok_s_per_user_p50"],r["acceptance"])
             for r in rows]
    def modeled(commits,prefill_speed,step_cost):
        gains=[(p+d)/(p/prefill_speed+d*base*step_cost/commits) for p,d,base in phases]
        new=projected_curve(*vectors(rows,gains),x,stock)
        return None if new is None else new*observed/reconstruct
    records=[]
    for ps in (1.,1.10,1.25,1.50):
        for step_cost in (.9,1.,1.1,1.25):
            ceiling=modeled(8.,ps,step_cost)
            record=dict(prefill_speed=ps,verify_step_cost_ratio=step_cost,
                        eight_commit_scenario=ceiling,required_mean_commits={})
            for goal in (1.8,2.):
                if ceiling is None or ceiling<goal:
                    record["required_mean_commits"][str(goal)]=None
                    continue
                lo,hi=1.,8.
                for _ in range(44):
                    mid=(lo+hi)/2
                    val=modeled(mid,ps,step_cost)
                    if val is not None and val>=goal:hi=mid
                    else:lo=mid
                record["required_mean_commits"][str(goal)]=hi
            records.append(record)
    out=dict(scope="Conditional sensitivity only. Rung phase durations use reported p90 first/later TTFT plus p50 generation. All generation time is optimistically scaled with changed verification count; all TTFT is scaled by the specified prefill factor. Interpolation assumes identical candidate speed movement on both axes. Equal absolute mean acceptance is assumed across loads, and width 8 caps it at 8. Cross-node effects, budget tails, queueing and measured tree overhead remain unresolved.",
             baseline_score=observed,baseline_reconstructed=reconstruct,
             phases=[dict(concurrency=r["concurrency"],prefill_proxy_s=p,decode_proxy_s=d,
                          decode_fraction=d/(p+d),incumbent_acceptance=c) for r,(p,d,c) in zip(rows,phases)],
             scenarios=records)
    (here/"frontier_requirements.json").write_text(json.dumps(out,indent=2))
    for r in records:
        if r["verify_step_cost_ratio"]==1.:print(r)
if __name__=="__main__":main()
