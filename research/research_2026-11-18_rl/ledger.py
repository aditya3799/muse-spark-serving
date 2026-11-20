"""Collect comparable local experiments and retain resolved shape-level regressions."""
import argparse
import importlib.util
import json
import math
import tempfile
from pathlib import Path

from score_trials import BASE, PLUGIN, ROOT

HERE = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE/"ledger.json")
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location("stats",PLUGIN)
    stats=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    files=list(HERE.glob("*_reward.json"))+list((ROOT/"research_2026-11-18_rl_prefill/gemm").glob("*_reward.json"))
    files+=list((ROOT/"research_2026-11-18_rl_dflash").glob("joint*_reward.json"))
    files+=list((HERE/"runs").glob("*/reward.json"))
    files+=list((HERE/"activation_search").glob("*_reward.json"))
    files+=list((ROOT/"research_2026-11-18_rl_experts/combined").glob("*_parent_reward.json"))
    files+=list((ROOT/"research_2026-11-18_rl_prefill/target_dense").glob("*_whole_reward.json"))
    files+=list((ROOT/"research_2026-11-18_rl_experts/narrow_n").glob("pilot_*_reward.json"))
    grouped=ROOT/"research_2026-11-18_rl_experts/grouped_route"
    # The independent confirmation receipt already includes the identical pilot.
    files+=[p for p in grouped.glob("*_reward.json") if p.name!="fused_reward.json" or not (grouped/"fused_held_reward.json").exists()]
    files+=list((ROOT/"research_2026-11-18_rl_experts/scale_expand").glob("*_reward.json"))
    for name in ("decode_pages/metadata_reward.json","decode_pages/global_rows/direct_reward.json",
                 "decode_pages/global_rows/suffix_reward.json","decode_qkv/entry_reward.json"):
        files.append(ROOT/"research_2026-11-18_rl_prefill"/name)
    files.append(ROOT/"research_2026-11-18_window_repair/decode_selector/selector_direct_walk_reward.json")
    for rel in ("decode_overlap/event_overlap_reward.json","bounded_reward.json"):
        files.append(ROOT/"research_2026-11-18_window_repair"/rel)
    for directory in ("feature_shard","placement","lattice_tree","decode_vocab","expert_slab","decode_snapshot"):
        files+=list((HERE/directory).glob("*_reward.json"))
    wide=ROOT/"research_2026-11-18_rl_experts/wide_n"
    for pilot in wide.glob("pilot_*_reward.json"):
        holdout=pilot.with_name(pilot.name.replace("pilot_", "holdout_", 1))
        files.append(holdout if holdout.exists() else pilot)
    entries=[]
    by_agent={}
    for path in sorted(set(files)):
        entry=json.loads(path.read_text(encoding="utf-8-sig"))
        if entry.get("baseline_sha256")!=BASE or "runs" not in entry:
            raise ValueError(path)
        trials=[]
        for run in entry["runs"]:
            row={k:run[k] for k in ("agent","candidate","scope","decision","reward","source")}
            for k in ("gain","ci","reason","source_sha256","candidate_sha256"):
                if k in run: row[k]=run[k]
            shapes={}
            for name, shape in run.get("shapes",{}).items():
                noise=run.get("noise")
                control=noise["shapes"][name] if noise else None
                observed=dict(shape,shapes={name:shape})
                null_control=dict(control,shapes={name:control}) if control else None
                decision,_=stats.decide(observed,noise=null_control)
                reward=100*math.log1p(shape["effect"]) if decision in ("KEEP","SLOWER") else 0.
                shapes[name]=dict(parent_us=shape["b_us"],candidate_us=shape["a_us"],
                                  gain=1+shape["effect"],decision=decision,reward=reward)
            row["shapes"]=shapes
            trials.append(row)
        last=trials[-1]
        verdict=entry.get("final_decision") or last["decision"]
        credit=entry.get("correctness_penalty",last["reward"])
        agent=last["agent"]
        account=by_agent.setdefault(agent,dict(rewarded=0,penalized=0,neutral=0,unmeasured=0,confirmed_keeps=0,shape_bonuses=0,shape_penalties=0))
        sign="unmeasured" if credit is None else "rewarded" if credit>0 else "penalized" if credit<0 else "neutral"
        account[sign]+=1
        account["confirmed_keeps"]+=int(entry["local_keep"])
        for shape in last["shapes"].values():
            account["shape_bonuses"]+=int(shape["reward"]>0)
            account["shape_penalties"]+=int(shape["reward"]<0)
        entries.append(dict(agent=agent,latest_verdict=verdict,latest_reward=credit,source=path.relative_to(ROOT).as_posix(),local_keep=entry["local_keep"],validation=entry.get("validation"),
                            final_decision=entry.get("final_decision"),correctness_penalty=entry.get("correctness_penalty"),runs=trials))
    receipt=dict(parent_sha256=BASE,official_parent_score=1.2363,hardware="RTX 3050 6 GB laptop, one rank",
                 unit="Signed rewards are 100 times log relative speed. Each scope has its own reference timing.",
                 agent_counts=by_agent,agent_counts_scope="Counts are per recorded experiment scope at its latest trial. Shape bonuses and penalties separately count resolved workload cells even when the overall measurement is noisy. Independent correctness vetoes override timing rewards. Gains in different scopes are never summed.",experiments=entries,excluded=json.loads((HERE/"excluded.json").read_text()))
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as tmp:
        json.dump(receipt,tmp,indent=2,allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(args.output)
    print(len(entries),"experiment scopes; records include positive, negative and inconclusive results")


if __name__=="__main__":main()
