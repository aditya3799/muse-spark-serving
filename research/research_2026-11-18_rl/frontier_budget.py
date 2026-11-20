"""Calculate necessary phase changes from the measured parent curve; never assign projected gains to unmeasured candidates."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"research_2026-11-18_frontier"))
from frontier_backtest import projected_curve,vectors


def main():
    p=json.loads((ROOT/"release_2026-11-18/scored_submission_status.json").read_text())["result"]
    rows=p["rungs"]; shape=p["shape"]
    anchors=[x["tok_s_per_user"] for x in p["frontier"]["points"]]
    stock=[x["stock_tok_s_per_gpu"] for x in p["frontier"]["points"]]
    phases=[[r["ttft_first_p90_s"],(shape["turns"]-1)*r["ttft_later_p90_s"],
             shape["turns"]*shape["output_mean"]/r["tok_s_per_user_p50"]] for r in rows]
    def est(first,later,decode):
        cost=[(a*first+b*later+c*decode) / (a+b+c) for a,b,c in phases]
        return projected_curve(*vectors(rows,[1/v for v in cost]),anchors,stock)
    thresholds=[]
    for target in (1.8,2.0):
        for f,l in ((1,1),(.75,1),(.5,1),(.75,.75),(.5,.5),(0,0)):
            low,hi=0.,1.
            if est(f,l,1)>=target:required=1.
            elif est(f,l,1e-8)<target:required=None
            else:
                for _ in range(60):
                    mid=(low+hi)/2
                    if est(f,l,mid)>=target:low=mid
                    else:hi=mid
                required=1/low
            thresholds.append(dict(target=target,first_duration_multiplier=f,later_duration_multiplier=l,required_decode_rate_gain=required))
    out=dict(scope="Counterfactual phase prioritization based on the official four-load frontier, p90 first and later TTFT and median generation. Assumes equal phase multipliers across loads. This is a model of required gains rather than measured future serving throughput.",
        phase_fraction=[[round(v/sum(t),5) for v in t] for t in phases],calibrated=est(1,1,1),
        best_if_first_turn_free=est(0,1,1),best_if_all_ttft_free=est(0,0,1),requirements=thresholds)
    (HERE/"frontier_budget.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out))

if __name__=="__main__":main()
