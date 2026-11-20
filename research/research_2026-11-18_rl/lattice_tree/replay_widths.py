"""Cross-fit an online draft-depth policy on disjoint requests in a real public GLM DFlash2 acceptance capture."""
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
from width_policy import OnlineAcceptance,choose_widths

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REV="9c64b393b4d5769ffb88d5d74a1f67f6985dea4f"
HF="https://huggingface.co/datasets/UW-SyFI/servingstudio-workload/resolve/"+REV+"/glm53_nvfp4/vllm_dflash2_k7/diverse_100/capture/"


def fetch(day,other_model=False):
    path=HERE/(("dspark_" if other_model else "")+day+"_acceptance.json")
    if not path.exists():
        base=HF if not other_model else HF.replace("glm53_nvfp4/vllm_dflash2_k7","deepseek_v4_flash/vllm_dspark_k7")
        url=base+day+"/"
        get=lambda f:urllib.request.urlopen(url+f,timeout=45).read()
        prov=json.loads(get("provenance.json"))
        trace=get("trace.csv")
        record=prov["files"]["trace.csv"]
        assert hashlib.sha256(trace).hexdigest()==record["sha256"]
        rows=list(csv.DictReader(io.StringIO(trace.decode())))
        assert len(rows)==100 and len({r["id"] for r in rows})==100
        data=dict(source=url+"trace.csv",sha256=record["sha256"],scope=record["scope"],
                  evidence=record["extra"]["accept_rate"],
                  requests=[dict(id=int(r["id"]),length=int(r["output_len"]),p=json.loads(r["accept_rate"])) for r in rows])
        path.write_text(json.dumps(data,indent=2))
    return json.loads(path.read_text())


def ref_curve():
    data=json.loads((ROOT/"profiles/725c19cbeb9b4d48.json").read_bytes())["benchmarks"]
    endpoints={}
    for b in (1,16,32):
        pairs=[]
        for w in (1,8):
            rows={r["case"]["layer"]:r for r in data if r["case"]["batch"]==b and r["case"].get("q_len",1)==w}
            assert set(rows)=={"sliding","full"}
            pairs.append((46*rows["sliding"]["baseline_ns"]+16*rows["full"]["baseline_ns"])/62)
        endpoints[b]=pairs
    return np.array([endpoints[k] for k in (1,16,32)])


def simulate(data,sub,load,policy,prior,extra,seed,rounded=True):
    chosen=[r for r in data["requests"] if r["id"]%2==sub]
    prob=np.array([r["p"] for r in chosen])
    ends=np.minimum(np.array([r["length"] for r in chosen]),220)
    n=len(ends)
    # Same random numbers for both arms at the same request and output position.
    rng=np.random.default_rng(seed)
    variates=rng.random((n,221,7))
    positions=np.zeros(n,dtype=int)
    actual=np.cumprod(variates<prob[:,None,:],axis=-1).sum(axis=-1)
    tracker=OnlineAcceptance(n,prior)
    order=rng.permutation(n); live=order[:load]; consumed=len(live)
    rows=steps=0; time=0.; curve=ref_curve()
    while len(live):
        b=len(live)
        a,c=[np.interp(b,[1,16,32],curve[:,i]) for i in range(2)]
        a,c=a+extra*c,c+extra*c
        remain=ends[live]-positions[live]
        if policy=="fixed":
            widths=np.repeat(8,b)
        elif policy=="budget_only":
            widths=np.minimum(8,remain)
        elif policy=="constant":
            widths=choose_widths(np.full((b,7),.75),remain,a,c,rounded)
        elif policy=="oracle":
            widths=choose_widths(prob[live],remain,a,c,rounded)
        else:
            widths=choose_widths(tracker.predict(live),remain,a,c,rounded)
            # A newly admitted request first sees the release's whole draft block.
            widths[positions[live]==0]=8
        total=int(widths.sum())
        paid=int(next((x for x in (b,2*b,4*b,8*b) if x>=total),8*b)) if rounded else total
        time+=a+(c-a)*(paid-b)/(7*b)
        accepted=np.minimum.reduce([actual[live,positions[live]],widths-1,remain-1])
        tracker.update(live,np.minimum(widths,remain),accepted)
        positions[live]+=1+accepted
        rows+=paid;steps+=1
        done=positions[live]==ends[live]
        keep=live[~done]; refill=order[consumed:consumed+int(done.sum())]
        consumed+=len(refill); live=np.r_[keep,refill]
    assert np.array_equal(positions,ends)
    return dict(rounds=steps,paid_rows=rows,total_tokens=int(ends.sum()),modeled_cost=float(time))


def main():
    train=held=fetch("20260922")
    # The September 20 replica contains no per-request acceptance column. Reserve odd IDs from the September 22 observation.
    def test(data,sub,seed,prior,extra,rounding):
        result={}
        for b in (1,4,16,32):
            arms={s:simulate(data,sub,b,s,prior,extra,seed,rounding) for s in ("fixed","budget_only","constant","online","oracle")}
            result[str(b)]=dict(arms=arms,online_ratio=arms["fixed"]["modeled_cost"]/arms["online"]["modeled_cost"],
                                oracle_ratio=arms["fixed"]["modeled_cost"]/arms["oracle"]["modeled_cost"],
                                online_vs_budget=arms["budget_only"]["modeled_cost"]/arms["online"]["modeled_cost"],
                                online_vs_constant=arms["constant"]["modeled_cost"]/arms["online"]["modeled_cost"])
        return result
    discoveries={p:test(train,0,991,p,.2,True) for p in (2,8,32)}
    best=max(discoveries,key=lambda p:np.mean([np.log(r["online_ratio"]) for r in discoveries[p].values()]))
    confirmations={str((extra,rounded)):test(held,1,7331,best,extra,rounded) for extra,rounded in ((0.,True),(.2,True),(.5,True),(.2,False))}
    independent=test(fetch("20260922",True),1,93177,best,.2,True)
    out=dict(external_dspark=independent,external_note="Independent DeepSeek-V4-Flash/DSpark seven-token capture, same source request texts and lengths as GLM. Challenges transfer across models, not across workload source.",
             scope="Public real per-request conditional success rates, independently resampled rounds, distinct GLM requests from one capture. Costs interpolate stock isolated H200 layer time and an assumed common draft share. Throughput is modeled; no new official run or target-model output is observed.",
             curve_ns_per_layer=ref_curve().tolist(),models="GLM-5.3-NVFP4 + GLM-5.3-DFlash2 (7 proposals)",
             shape="Independent requests capped at 220 remaining output tokens, continuously refilled, rounded catalog per live batch 1,2,4,8 average rows",
             prior=best,discovery=discoveries,holdout=confirmations)
    (HERE/"width_search.json").write_text(json.dumps(out,indent=2))
    print("prior",best,"holdout online/oracle",[(r["online_ratio"],r["oracle_ratio"]) for r in confirmations[str((.2,True))].values()])


if __name__=="__main__":
    main()