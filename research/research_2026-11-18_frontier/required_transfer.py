"""Invert the modeled frontier to identify which unmeasured assumption carries it."""
import json
import math
from pathlib import Path
from frontier_backtest import projected_curve,vectors

HERE=Path(__file__).resolve().parent
S=HERE.parent


def main():
    parent=json.loads((S/"release_2026-11-18/scored_submission_status.json").read_text())["result"]
    sensitivity=json.loads((S/"research_2026-11-18_expert_tp/target_sensitivity.json").read_text())
    p=parent["frontier"]["points"]
    anchors=[r["tok_s_per_user"] for r in p]
    stock=[r["stock_tok_s_per_gpu"] for r in p]
    fractions=next(m for m in sensitivity["models"] if m["transfer"]==.5 and m["net_added_us_per_seam"]==2)
    savings={int(k):v["saving_ms"] for k,v in sensitivity["layer"].items()}
    savings[4]=savings[1]+.2*(savings[16]-savings[1])
    def fraction_curve(transfer,decode_copy):
        gains=[]
        for load,r in zip(fractions["loads"],parent["rungs"]):
            # Both ends of each of 62 layers can add a synchronization cost.
            cycle=1000*r["acceptance"]/r["tok_s_per_user_p50"]
            new_cycle=cycle-transfer*savings[r["concurrency"]]+.248
            if new_cycle<=0:
                raise ValueError("unphysical per-verify runtime")
            gain=1/(load["first_fraction"]/fractions["first_gain"]+
                    load["later_fraction"]/fractions["later_gain"]+
                    load["generation_fraction"]*new_cycle/(cycle*decode_copy))
            gains.append(gain)
        return projected_curve(*vectors(parent["rungs"],gains),anchors,stock)
    results=[]
    for corpus in ("swesmith","openhands","muse_joined","muse_split"):
        replay=json.loads((HERE/f"{corpus}_muse_replay.json").read_text())
        for r in replay["results"]:
            if not r["matched_four_commits"]:
                continue
            for cost in ("1.25","2"):
                for proxy in ("point","bootstrap_low"):
                    g=r["wide_cost_sensitivity"][cost]
                    decode_copy=g["ratio"] if proxy=="point" else g["trace_interval"][0]
                    lo,hi=0.,1.
                    possible=fraction_curve(hi,decode_copy)>=1.8
                    if possible:
                        for _ in range(40):
                            t=(lo+hi)/2
                            if fraction_curve(t,decode_copy)>=1.8:
                                hi=t
                            else:
                                lo=t
                    results.append(dict(corpus=corpus,conditional_neural_commits=r["assumed_fallback_commit_on_copy_opportunity"],
                                        wide_cost=float(cost),proxy=proxy,copy_decode=decode_copy,
                                        required_donor_transfer=hi if possible else None,
                                        modeled_at_full_transfer=fraction_curve(1,decode_copy)))
    report=dict(results=results,
      scope="Inversion assumes median isolated prefill gains, 2us added at two seams per layer, proportional transfer of one-device decoder savings, common copy gain at every concurrency, and maintained interactivity curve. The bootstrap low endpoint reflects only trajectory mix. Required transfer has no assigned probability.")
    (HERE/"required_transfer.json").write_text(json.dumps(report,indent=2))
    print([(r["corpus"],r["proxy"],r["required_donor_transfer"]) for r in results
           if r["conditional_neural_commits"]==4 and r["wide_cost"]==1.25])


if __name__=="__main__":
    main()
