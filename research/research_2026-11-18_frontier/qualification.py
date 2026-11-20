"""Reconcile paired GPU timings, official matched frontier and public replay sensitivities."""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVING = HERE.parent
PLUGIN = SERVING.parent/"scratch/kar-serving-research"
sys.path.insert(0,str(PLUGIN))
from kar.evaluate.stats import compare, decide


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    shapes = {}
    sizes = []
    for name in ("gpu_abba.json","gpu_cold_prefill.json"):
        data = read(SERVING/"research_2026-11-18_prefill_route_windows"/name)
        for case in data["cases"]:
            key = f'{case["pattern"]}_{case["m"]}'
            sizes.append(case["m"])
            shapes[key] = dict(weight=1,times={
                label:[1000*t for t in case[samples]]
                for label,samples in (("binned","ms_windows"),("reference","ms_reference"))})
    paired = compare(dict(shapes=shapes,instances=1),["binned"],["reference"])
    verdict,why=decide(paired)
    sensitivity = read(SERVING/"research_2026-11-18_expert_tp/target_sensitivity.json")
    parent = read(SERVING/"release_2026-11-18/scored_submission_status.json")["result"]["frontier"]
    stock = {p["concurrency"]:p["ratio"] for p in parent["points"]}
    combined=[]
    for corpus in ("swesmith","openhands","muse_joined","muse_split"):
        replay = read(HERE/f"{corpus}_muse_replay.json")
        for row in replay["results"]:
            if not row["matched_four_commits"]:
                continue
            for transfer in (0,.5,1):
                model = next(m for m in sensitivity["models"] if m["transfer"]==max(transfer,.5)
                             and m["net_added_us_per_seam"]==2 and m["c4_interpolation"]==.2)
                for cost in ("1.25","2"):
                    incremental = row["wide_cost_sensitivity"][cost]["ratio"]
                    ratios = []
                    for load in model["loads"]:
                        fg = model["first_gain"] if transfer else 1
                        lg = model["later_gain"] if transfer else 1
                        dg = load["decode_gain_proxy"] if transfer else 1
                        # This requires the modeled change in cost per emitted token
                        # to survive at every concurrency after target improvements.
                        new_time=(load["first_fraction"]/fg+load["later_fraction"]/lg+
                                  load["generation_fraction"]/(dg*incremental))
                        ratios.append(stock[load["users"]]/new_time)
                    combined.append(dict(corpus=corpus,assumed_copy_site_neural_commits=row["assumed_fallback_commit_on_copy_opportunity"],
                        assumed_wide_cost=float(cost),assumed_target_saving_transfer=transfer,
                        decode_replay=incremental,load_ratios=ratios,geomean=math.prod(ratios)**.25))
    report=dict(measured_frontier=parent["score"],
                plugin_prefill=dict(verdict=verdict,reason=why,detail=paired,
                    scope="One placement, one or two AB/BA pairs per shape. Equal weights across nine synthetic SM86 shapes. Within-fixture descriptive effect only."),
                portfolio_scenarios=combined,
                limits="Each serves a sensitivity, with synthetic neural accept lengths for public coding traces and an assumed wide verifier cost. Joint scenarios assume the copy improvement transfers to every load and the faster target, on top of median prefill layer gains. They do not model frontier interpolation, competition outputs or measured distributed latency.")
    (HERE/"qualification.json").write_text(json.dumps(report,indent=2))
    print(verdict,why,parent["score"])
    for row in combined:
        if row["assumed_copy_site_neural_commits"]==4 and row["assumed_wide_cost"]==1.25:
            print(row["corpus"],row["assumed_target_saving_transfer"],round(row["geomean"],3))


if __name__ == "__main__":
    main()
