"""Compare frozen target vocabulary projection, output scaling, softcap and greedy selection on one TP8 shard."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch
from sharded_max import parallel_softcap_max

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/"release_2026-11-18"
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
S=ROOT/"sglang/python/sglang"
path=S/"kernels/ops/sampling/softcap_row_argmax.py"
spec=importlib.util.spec_from_file_location("parent_decode_softcap",path)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
model=PARENT/"candidate_sources/python/sglang/srt/models/muse_spark.py"
if not model.exists():model=S/"srt/models/muse_spark.py"
tree=ast.parse(model.read_bytes())
cls=next(n for n in tree.body if getattr(n,"name","")=="MuseSpark1xLogitsProcessor")
method=next(n for n in cls.body if getattr(n,"name","")=="_compute_lm_head")
scope={"torch":torch,"use_intel_amx_backend":lambda t:False}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(model),"exec"),scope)
from types import SimpleNamespace
CFG=SimpleNamespace(use_fp32_output=True)
SCALE=.1767766952966369
WIDTH=25256


def fixed_cases(seed):
    g=torch.Generator(device="cuda").manual_seed(seed); checks=[]
    for n in (0,1,4095,4096,4097,25256):
        for kind in ("random","ties","plateau","special"):
            storage=torch.randn((8,WIDTH+11),device="cuda",generator=g)
            x=storage[:,:WIDTH]
            if kind=="ties":
                x.fill_(-120);x[:,::2048]=80
            elif kind=="plateau":
                x[:]=(torch.randn(x.shape,device="cuda",generator=g)*.005+1340)
            elif kind=="special":
                x[:,::510]=float("nan")
                x[0]=float("-inf")
                x[1]=float("nan")
                x[2,::4095]=float("inf")
            if n<WIDTH:x[:,n:]=float("inf")
            for scale in (1.,SCALE):
                orig=parent.softcap_row_argmax(x*scale,20.,n,3*WIDTH)
                test=parallel_softcap_max(x,20.,n,3*WIDTH,scale=scale)
                checks.append(dict(width=n,kind=kind,scale=scale,exact=torch.equal(orig.view(torch.int32),test.view(torch.int32))))
    return checks


def capture(op):
    op();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result=op()
    return graph,result


def time(graph):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();graph.replay();end.record();end.synchronize()
    return begin.elapsed_time(end)*1000


def test(m,head,seed,rounds,full):
    g=torch.Generator(device="cuda").manual_seed(seed)
    x=torch.empty(m,8192,dtype=torch.bfloat16,device="cuda")
    fixed=torch.empty(m,WIDTH,device="cuda")
    x.normal_(generator=g);fixed.normal_(generator=g)
    def run(early):
        # Unchanged target GEMM, same BF16 operands and FP32 accumulator boundary.
        logits=scope["_compute_lm_head"](CFG,x,head) if full else fixed
        if early:return parallel_softcap_max(logits,20.,WIDTH,0,scale=SCALE)
        scaled=logits.mul_(SCALE) if full else torch.mul(logits,SCALE)
        return parent.softcap_row_argmax(scaled,20.,WIDTH,0)
    graphs={name:capture(lambda fast=(name=="candidate"):run(fast)) for name in ("baseline","candidate")}
    arm_order=("baseline","candidate","control")
    times={n:[] for n in arm_order}
    match=[];updated=[];previous=None
    for i in range(rounds):
        if full:x.normal_(std=.75,generator=g)
        else:fixed.normal_(std=2.,generator=g)
        for arm in (arm_order if i%2==0 else arm_order[::-1]):
            times[arm].append(time(graphs["baseline" if arm=="control" else arm][0]))
        a=graphs["baseline"][1];b=graphs["candidate"][1]
        match.append(torch.equal(a.view(torch.int32),b.view(torch.int32)))
        updated.append(previous is None or not torch.equal(a,previous))
        previous=a.clone()
    return dict(weight=1,times=times,exact=match,changing=updated)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--rounds",type=int,default=8);p.add_argument("--loads",type=int,nargs="+",default=[1,4,16,32])
    p.add_argument("--out",type=Path,required=True);p.add_argument("--only",choices=("both","vocab","head"),default="both")
    a=p.parse_args();torch.manual_seed(a.seed)
    assert hashlib.sha256((PARENT/"candidate.diff").read_bytes()).hexdigest()==BASE
    head=SimpleNamespace(weight=torch.empty(WIDTH,8192,dtype=torch.bfloat16,device="cuda").normal_(std=.1))
    checks=fixed_cases(a.seed)
    print("edge cases",sum(r["exact"] for r in checks),"/",len(checks),flush=True)
    report=dict(baseline_sha256=BASE,candidate_sha256=hashlib.sha256((HERE/"sharded_max.py").read_bytes()).hexdigest(),
                agent="decode-vocabulary",candidate="parallel-scaled-softcap",seed=a.seed,active=True,complete=False,correct=False,
                scope="Rank-local greedy target vocab shard in DFlash verify; genuine scored source, synthetic BF16 head/activations, 8192 hidden and 25256 shard vocab. No cross-rank selection cost.",checks=checks,
                projection_authority=str(model.relative_to(ROOT)),bench={"instances":1,"shapes":{}})
    a.out.write_text(json.dumps(report,indent=2))
    for full in (False,True):
        if a.only!="both" and full!=(a.only=="head"):continue
        for load in a.loads:
            m=8*load; key=("head" if full else "vocab")+str(load)
            value=test(m,head,a.seed+31*load,a.rounds,full);report["bench"]["shapes"][key]=value
            med=lambda x:float(torch.tensor(x,dtype=torch.float64).median())
            print(key,med(value["times"]["baseline"])/med(value["times"]["candidate"]),all(value["exact"]),flush=True)
            a.out.write_text(json.dumps(report,indent=2))
    report["correct"]=all(c["exact"] for c in checks) and all(all(r["exact"]) and all(r["changing"]) for r in report["bench"]["shapes"].values())
    report["complete"]=True;a.out.write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    with torch.inference_mode():main()
