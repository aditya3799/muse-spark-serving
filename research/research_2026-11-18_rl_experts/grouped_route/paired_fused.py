"""Paired complete local FFN, changed routes for each AB/BA and A/A."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
base=(here.parent/"narrow_n"/"paired.py").read_text()
setup=base.split("parser=argparse.ArgumentParser()")[0]
exec(compile(setup,str(here/"paired.py"),"exec"),globals())
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--seed",type=int,required=True)
parser.add_argument("--rows",type=int,nargs="+",required=True)
parser.add_argument("--variant",choices=("split_fused","single_light","single_heavy"),default="split_fused")
parser.add_argument("--rounds",type=int,default=8)
args=parser.parse_args()
weights=base[base.index("raw=torch.nn.Module();"):base.index("from policy import narrow")]
def finite_scales(e,n,h):
    v=torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8)
    codes=torch.tensor([i for i in range(256) if i not in (127,255)],device="cuda",dtype=torch.uint8)
    v[:,:254,0].copy_(codes)
    return v.view(torch.float8_e4m3fn)
weights=weights.replace('torch.randint(40,68,(e,n,h//16),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)',"finite_scales(e,n,h)")
exec(compile(weights,str(here/"paired.py"),"exec"),globals())
import grouped_fused as grouped
sha=hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest()
assert sha=="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
report=dict(agent="experts",candidate="stable_expert_major_grouped_"+args.variant,scope="Complete rank-local FFN, one RTX3050, kernel-task EP8 busiest-rank routes, peer pool partials prestaged, no collectives/attention",baseline_sha256=sha,candidate_sha256=hashlib.sha256((here/"grouped_fused.py").read_bytes()+args.variant.encode()).hexdigest(),seed=args.seed,complete=False,correct=True,active=False,unit="us",bench=dict(instances=1,shapes={}),failures=[],route_checks=[],dispatch=[])
def save():
    tmp=args.out.with_suffix(".temp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.out)
original=HummingMethod.forward_layer
def dispatch(layer,**kw):
    if kw["compute_config"]["gemm_type"]=="grouped_contiguous":
        entry=dict(stage=kw["sublayer_name"],rows=kw["valid_shape_m"],block=kw["tuning_config"]["block_shape"],warp=kw["tuning_config"]["warp_shape"])
        if entry not in report["dispatch"]:report["dispatch"].append(entry)
    return original(layer,**kw)
HummingMethod.forward_layer=staticmethod(dispatch)
def case(m):
    env=DenseEnvelope(m,args.seed)
    x=torch.empty((m,8192),device="cuda",dtype=torch.bfloat16)
    ids=torch.empty((m,8),device="cuda",dtype=torch.int32)
    scores=torch.empty((m,8),device="cuda")
    def update(seed,dist):
        data=ref.generate_input(m,"ep8",seed,dist="normal" if dist.startswith("all_") else dist,rank="busiest",device="cuda")
        raw=data["topk_ids"]-data["config"]["local_start"]
        ids.copy_(torch.where((raw>=0)&(raw<32),raw,-1))
        if dist=="all_remote":ids.fill_(-1)
        if dist=="all_local":ids.copy_((torch.arange(8,device="cuda")[None]+torch.arange(m,device="cuda")[:,None])%32)
        scores.copy_(data["topk_weights"]);torch.manual_seed(seed)
        x.normal_(std=.023);env.stage_peers(env.peer_fixture(x))
    def run(candidate):
        inner=env.before(x)
        if candidate:
            return env.after(grouped.compute(layer,inner,ids,split,args.variant,scores,norm))
        else:
            y,_=ns["_humming_expert_outputs"](layer,inner,ids,torch,global_experts=256,defer_downscale=True)
        val=finish(y,scores,norm_weight=norm,eps=1e-8,weight_first=True,slot_expert_ids=ids,expert_output_scale=layer.w2_post_scale)
        return env.after(val)
    def compare(a,b,dist):
        exact=torch.equal(a.view(torch.int16),b.view(torch.int16))
        if not exact:
            delta=(a.float()-b.float()).abs()
            report.update(correct=False,active=True,complete=True)
            report["failures"].append(dict(m=m,dist=dist,n=int(torch.count_nonzero(a.view(torch.int16)!=b.view(torch.int16))),max=float(delta.max()),finite=bool(torch.isfinite(b).all())))
            save();print("PARITY FAILURE",report["failures"][-1],flush=True);sys.exit(3)
        return True
    update(args.seed,"normal")
    graphs=[];outputs=[]
    for cand in (False,True):
        run(cand);torch.cuda.synchronize()
        with torch.cuda.graph(g:=torch.cuda.CUDAGraph()):outputs.append(run(cand))
        graphs.append(g);g.replay()
    report["active"]=True
    for dist in ("all_remote","all_local","normal","skewed"):
        update(args.seed+111,dist)
        configs=split._choose_large_prefill_configs(layer,m*8,256)
        cutoff=configs["cutoff"] if configs and args.variant=="split_fused" else -1
        ranks,offsets,light=grouped.mapping(ids,cutoff)
        hostids=ids.flatten().tolist();hostmap=ranks.tolist();hostlayout=offsets.tolist()
        for expert in range(32):
            indices=[j for j,k in enumerate(hostids) if k==expert]
            islight=len(indices)<=cutoff
            loc=hostlayout[(0 if islight else 33)+expert]+(0 if islight else light)
            assert [hostmap[j] for j in indices]==list(range(loc,loc+len(indices)))
        placed=[j for j in hostmap if j>=0];assert len(set(placed))==len(placed)
        assert all((a<0)==(b<0) for a,b in zip(hostids,hostmap))
        graphs[0].replay();graphs[1].replay()
        report["route_checks"].append(dict(rows=m,dist=dist,placed=len(placed),exact=compare(*outputs,dist)))
    for dist in ("normal","skewed"):
        name=f"M{m}_{dist}";r=report["bench"]["shapes"][name]=dict(weight=1,times={k:[] for k in ("baseline","candidate","control")},orders=[],parity=[])
        for j in range(args.rounds):
            update(args.seed+997*m+97*j+(7717 if dist=="skewed" else 0),dist)
            order=["baseline","candidate","control"] if j%2==0 else ["control","candidate","baseline"];r["orders"].append(order)
            for k in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();graphs[int(k=="candidate")].replay();b.record();b.synchronize();r["times"][k].append(a.elapsed_time(b)*1000)
            r["parity"].append(compare(*outputs,dist));save()
        print(name,statistics.median(r["times"]["baseline"])/statistics.median(r["times"]["candidate"]),flush=True)
for m in args.rows:
    case(m);torch.cuda.empty_cache()
report["complete"]=True;save()
