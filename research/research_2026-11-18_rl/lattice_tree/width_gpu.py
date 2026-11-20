"""Run variable-width target verification through the frozen Humming expert kernels at full H=I=4096."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"placement"))
from worker import native,torch,BASE,fixture,finish,cached_threshold_for_preparation,timed
from width_policy import OnlineAcceptance,choose_widths
from replay_widths import ref_curve


def layer_from_release():
    raw=torch.nn.Module()
    for stage,n in (("w13",8192),("w2",4096)):
        for name,buf in (("weight",torch.randint(256,(32,n,2048),dtype=torch.uint8,device="cuda")),
                        ("weight_scale",torch.randint(40,68,(32,n,256),dtype=torch.uint8,device="cuda").view(torch.float8_e4m3fn)),
                        ("weight_scale_2",torch.linspace(.125,.48,32,device="cuda",dtype=torch.bfloat16))):
            raw.register_parameter(stage+"_"+name,torch.nn.Parameter(buf,requires_grad=False))
    with cached_threshold_for_preparation(HERE):
        return native["_prepare_humming_experts"](raw,torch)


def evaluate(layer,b,seed,rounds,kind):
    rng=np.random.default_rng(seed)
    reqs=json.loads((HERE/"20260922_acceptance.json").read_text())["requests"]
    held=[r for r in reqs if r["id"]%2 and r["length"]>=48]
    req_idx=rng.choice(len(held),b,replace=b>len(held))
    probabilities=np.array([held[j]["p"] for j in req_idx])
    track=OnlineAcceptance(b,8)
    endpoint=ref_curve()
    a,c=[np.interp(b,[1,16,32],endpoint[:,i]) for i in range(2)]
    cmin,cmax=a+.2*c,c+.2*c
    # Burn-in is request history of preceding width-eight queries, independent of test activations.
    for _ in range(12):
        count=np.cumprod(rng.random((b,7))<probabilities,axis=1).sum(1)
        track.update(np.arange(b),np.full(b,8),count)
    input_x=torch.empty((8*b,4096),device="cuda",dtype=torch.bfloat16)
    input_ids=torch.empty(8*b,8,device="cuda",dtype=torch.int32)
    input_w=torch.empty((8*b,8),device="cuda")
    norm=torch.linspace(.7,1.15,4096,device="cuda",dtype=torch.bfloat16)
    def fill(i):
        gen=torch.Generator(device="cuda").manual_seed(seed+997*i)
        input_x.normal_(std=.023,generator=gen)
        keys=torch.rand((8*b,256),generator=gen,device="cuda")
        if kind=="skewed":keys[::2,:8]+=1.
        ids=keys.topk(8).indices
        input_ids.copy_(torch.where(ids<32,ids,-1));input_w.uniform_(generator=gen);input_w.div_(input_w.sum(-1,keepdim=True))
    def run(x,ids,s):
        down,live=native["_humming_expert_outputs"](layer,x,ids,torch,defer_downscale=True,global_experts=256)
        return finish(down,s,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=live,expert_output_scale=layer.w2_post_scale)
    fill(0)
    run(input_x,input_ids,input_w);torch.cuda.synchronize()
    parent_graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(parent_graph):
        parent_out=run(input_x,input_ids,input_w)
    captures={}
    def variant(bucket):
        if bucket not in captures:
            rows=torch.arange(bucket,device="cuda",dtype=torch.long)
            live=torch.ones(bucket,device="cuda",dtype=torch.bool)
            def compact():
                # The patched method receives packed offsets from the verify planner before target kernels.
                ids=torch.where(live[:,None],input_ids.index_select(0,rows),-1)
                return run(input_x.index_select(0,rows),ids,input_w.index_select(0,rows))
            compact();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=compact()
            captures[bucket]=(graph,out,rows,live)
        return captures[bucket]
    results=[]
    for i in range(rounds):
        widths=choose_widths(track.predict(np.arange(b)),np.full(b,200-i),cmin,cmax)
        budget=next(v for v in (b,2*b,4*b,8*b) if v>=widths.sum())
        graph,out,indices,live=variant(budget)
        chosen=np.concatenate([np.arange(8*r,8*r+w) for r,w in enumerate(widths)])
        count=len(chosen)
        index=torch.from_numpy(np.pad(chosen,(0,budget-count))).to("cuda")
        indices.copy_(index);live.copy_(torch.arange(budget,device="cuda")<count)
        fill(i+1)
        arms=("baseline","candidate","control")
        times={k:timed(graph if k=="candidate" else parent_graph) for k in (arms if i%2==0 else arms[::-1])}
        ref=parent_out.index_select(0,index[:count])
        diff=ref.float()-out[:count].float()
        accepted=np.cumprod(rng.random((b,7))<probabilities,axis=1).sum(1)
        actual=np.minimum(accepted,widths-1)
        results.append(dict(timing_us=times,physical_rows=budget,live_rows=count,widths=widths.tolist(),
            ref_commits_proxy=int((1+accepted).sum()),candidate_commits_proxy=int((1+actual).sum()),
            bit_mismatches=int((ref.view(torch.int16)!=out[:count].view(torch.int16)).sum()),
            relative_l2=float(diff.norm()/ref.float().norm().clamp_min(1e-10)),finite=bool(torch.isfinite(out[:count]).all())))
        track.update(np.arange(b),widths,actual)
    return dict(request_ids=[held[j]["id"] for j in req_idx],rows=results)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--loads",nargs="+",type=int,default=[16,32]);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--rounds",type=int,default=8);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--patterns",nargs="+",default=["uniform","skewed"]);a=p.parse_args()
    assert hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest()==BASE
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    layer=layer_from_release()
    report=dict(baseline=BASE,seed=a.seed,complete=False,shapes={},
        scope="Identical source activations for one true frozen 4096-wide Humming expert pair and native finalization; candidate includes row packing. Selection from simulated prior observations of real GLM rates, not native Muse acceptances. Whole target attention, draft and interconnect excluded.")
    for b in a.loads:
        for pattern in a.patterns:
            value=evaluate(layer,b,a.seed,a.rounds,pattern)
            report["shapes"][f"B{b}_{pattern}"]=value
            print(b,pattern,[r["physical_rows"] for r in value["rows"]],sum(r["bit_mismatches"] for r in value["rows"]),flush=True)
            a.out.write_text(json.dumps(report,indent=2))
    report["complete"]=True
    a.out.write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    with torch.inference_mode():main()
