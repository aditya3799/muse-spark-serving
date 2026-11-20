"""Historical counterfactual of elapsed-phase estimates and frontier interpolation.

Estimate each later scored run using its actual aggregate TTFT/generation ratios
and an earlier run's observed raw throughput curve. The later raw rates are used
only to grade the estimate. Different prompts/nodes can change these summaries.
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"research_2026-11-18_expert_tp"))
from frontier_calibration import refs


def projected_curve(x,y,anchors,stock):
    points=[]
    if any(x[i] <= x[i+1] for i in range(len(x)-1)):
        return None
    for t,s in zip(anchors,stock):
        if t > x[0]*1.00001:
            return None
        ry=y[-1]
        for i in range(len(x)-1):
            if x[i]>=t>=x[i+1]:
                a=math.log(t/x[i])/math.log(x[i+1]/x[i])
                ry=y[i]*(y[i+1]/y[i])**a
                break
        points.append(ry/s)
    return statistics.geometric_mean(points)


def duration(r,shape):
    turns=shape["turns"]
    return (r["ttft_first_p90_s"]+(turns-1)*r["ttft_later_p90_s"]+
            turns*shape["output_mean"]/r["tok_s_per_user_p50"])


def vectors(parent, gains):
    return ([r["tok_s_per_user_e2e"]*g for r,g in zip(parent,gains)],
            [r["tok_s_per_gpu"]*g for r,g in zip(parent,gains)])


def main():
    runs=[]
    for info in refs()[0]:
        receipt=json.loads((HERE.parent/info["path"]).read_bytes())
        receipt=receipt.get("response",receipt)
        runs.append((receipt["id"],receipt["result"]))
    comparisons=[]
    for i,(job,new) in enumerate(runs):
        pts=new["frontier"]["points"]
        anchors=[p["tok_s_per_user"] for p in pts]
        stock=[p["stock_tok_s_per_gpu"] for p in pts]
        for previous,old in runs[:i]:
            ratios=[duration(before,old["shape"])/duration(after,new["shape"])
                    for before,after in zip(old["rungs"],new["rungs"])]
            x,y=vectors(old["rungs"],ratios)
            score=projected_curve(x,y,anchors,stock)
            comparisons.append(dict(before=previous,after=job,modeled=score,
                official=new["frontier"]["score"],raw_gpu_errors=[a/b["tok_s_per_gpu"]-1 for a,b in zip(y,new["rungs"])],
                relative_error=None if score is None else score/new["frontier"]["score"]-1))
    parent=runs[-1][1]
    points=parent["frontier"]["points"]
    x_fixed=[p["tok_s_per_user"] for p in points]
    y_fixed=[p["stock_tok_s_per_gpu"] for p in points]
    parent_reconstructed=projected_curve(*vectors(parent["rungs"],[1]*4),x_fixed,y_fixed)
    assert abs(parent_reconstructed/parent["frontier"]["score"]-1)<.02
    predictions=[]
    for scenario in json.loads((HERE/"qualification.json").read_text())["portfolio_scenarios"]:
        gains=[y/p["ratio"] for y,p in zip(scenario["load_ratios"],points)]
        curve_score=projected_curve(*vectors(parent["rungs"],gains),x_fixed,y_fixed)
        predictions.append(dict(corpus=scenario["corpus"],assumed_transfer=scenario["assumed_target_saving_transfer"],
                            assumed_wide_cost=scenario["assumed_wide_cost"],conditional_acceptance=scenario["assumed_copy_site_neural_commits"],
                            direct_score=scenario["geomean"],curve_score=curve_score))
    errors=[abs(p["relative_error"]) for p in comparisons if p["relative_error"] is not None]
    report=dict(backtests=comparisons,parent_curve_reconstruction=parent_reconstructed,
        covered_pairs=len(errors),requested_pairs=len(comparisons),mean_absolute_relative_error=statistics.mean(errors),
        max_absolute_relative_error=max(errors),sensitivity=predictions,
        scope="Synthetic future scores assume identical time changes in wall throughput and user speed, preserved curve shape between rungs, saturation at highest measured load, and cross-corpus decode transfer. Backtests cover related historical candidate changes only. Component transfer is unmeasured.")
    (HERE/"frontier_backtest.json").write_text(json.dumps(report,indent=2))
    print("covered",len(errors),"of",len(comparisons),"mean abs error",statistics.mean(errors),"max",max(errors))


if __name__=="__main__":
    main()

