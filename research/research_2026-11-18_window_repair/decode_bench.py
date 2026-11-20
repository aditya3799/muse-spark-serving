"""Captured _SelectorDraftSampler A/B after real global candidate selection or after lattice scoring."""
from __future__ import annotations
import ast,__future__,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace as S,MethodType
import torch
H=Path(__file__).resolve().parent;D=H/"decode_selector";sys.path.insert(0,str(D))
import direct_walk,original_walk
SEEDS={"pilot":813733,"holdout":295817}
ARMS=("baseline","candidate","control")
BASE="8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
def source_node(source,name,ns,cl=None):
    owner=ast.parse(source)
    if cl:owner=next(x for x in owner.body if getattr(x,"name",None)==cl)
    node=next(x for x in owner.body if getattr(x,"name",None)==name)
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(D/name),"exec",flags=__future__.annotations.compiler_flag),ns)
    return ns[name]
def prepare(seed,mode):
    g=torch.Generator(device="cuda").manual_seed(seed)
    def rand(shape,std=1):return torch.empty(shape,device="cuda",dtype=torch.bfloat16).normal_(std=std,generator=g)
    # Public checkpoint dimensions. Global top-k IDs are inputs, so no fake TP gather.
    v=202048;r=256;h=8192
    weights=(rand((v,r),.2),rand((v,r),.2),rand((r,h),h**-.5)) if mode!="walk" else (None,None,None)
    model=(D/"dflash_original_src.py").read_text(encoding="utf8")
    return g,weights,model
def run(b,seed,stage,scope,g,weights,model,rounds=16):
    B,SLOTS,K,HIDDEN=b,7,16,8192
    device="cuda";greedy=scope=="greedy"
    ids=torch.randint(0,202048,(b,SLOTS,K),device=device,generator=g,dtype=torch.int64)
    unary=torch.randn((b,SLOTS,K),device=device,generator=g)
    scores=torch.randn((b,SLOTS,K,K),device=device,generator=g)
    hs=torch.randn((b*8,HIDDEN),device=device,generator=g,dtype=torch.bfloat16)
    input_ids=torch.randint(0,202048,(b*8,),device=device,generator=g)
    original=(D/"dflash_worker_v2_original_src.py").read_text(encoding="utf8");direct=(D/"dflash_worker_v2_direct_src.py").read_text(encoding="utf8")
    arms={};graphs={}
    edges=source_node(model,"_score_edges",dict(torch=torch))
    select_fn=source_node(model,"build_lattice",dict(torch=torch,_score_edges=edges),cl="CandidateSelector")
    for arm in ARMS:
        native=original_walk if arm!="candidate" else direct_walk
        selector=S(top_k=K,predecessor_codebook=weights[0],successor_codebook=weights[1],hidden_projection=lambda v:torch.nn.functional.linear(v,weights[2]))
        selector.build_lattice=MethodType(select_fn,selector)
        selector.sample_path=native.selector_walk_triton
        if arm=="candidate":selector.sample_path_into=native.selector_walk_triton
        draft=S(candidate_selector=selector,compute_candidates=lambda hidden:(ids.view(-1,K),unary.view(-1,K)))
        src=original if arm!="candidate" else direct
        lattice=source_node(src,"_selector_lattice",{})
        if stage=="walk":lattice=lambda *args:(ids,scores)
        sampler_cls=source_node(src,"_SelectorDraftSampler",dict(torch=torch,_selector_lattice=lattice,
            resolve_greedy_mask=lambda *,bs,sampling_info,device:(sampling_info.top_ks<=1).view(-1)))
        sampler=sampler_cls(draft_model=draft,block_size=8,max_bs=b,device=device,sampling_enabled=True)
        top=torch.ones(b,device=device,dtype=torch.int32)
        if not greedy:top[::2]=8
        sampling=S(temperatures=torch.ones(b,device=device),top_ks=top)
        sampler.stage_sampling_params(bs=b,sampling_info=sampling)
        with torch.cuda.stream(torch.cuda.Stream()):sampler(hs,input_ids)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):sampler(hs,input_ids)
        arms[arm]=sampler;graphs[arm]=graph
    times={a:[] for a in ARMS};orders=[];wrong={a:dict(tokens=0,q=0,ids=0) for a in ARMS[1:]};changed=[]
    last=None
    for step in range(rounds):
        ids.add_(67).remainder_(202048);unary.copy_(unary.roll(1,-1));scores.copy_(scores.roll(1,-1))
        if stage=="walk" and step in (1,3) and greedy:
            scores[:,:,0,:2]=9
            if step==3:scores[:,0,0,4]=float("nan")
        for a in ARMS:
            mask=arms[a].greedy_mask
            mask.fill_(True)
            if not greedy:mask[step%2::2]=False
            if b==1 and not greedy:mask.zero_()
            arms[a].temperatures.fill_(.53 if step%2 else 1.37)
        torch.cuda.synchronize()
        order=list(ARMS[step//2%3:]+ARMS[:step//2%3]);order.reverse() if step%2 else None
        orders.append(order)
        for a in order:
            torch.cuda.manual_seed(seed+3001*step+b)
            graphs[a].replay()
        torch.cuda.synchronize()
        ref=arms["baseline"]
        for a in ARMS[1:]:
            for field,name in (("tokens","out"),("q","q_out"),("ids","candidate_out")):
                x=getattr(ref,name);y=getattr(arms[a],name)
                if field=="q":x=x.view(torch.int32);y=y.view(torch.int32)
                wrong[a][field]+=torch.count_nonzero(x!=y).item()
        if last is not None:changed.append(torch.count_nonzero(ref.out!=last).item())
        last=ref.out.clone()
        for a in order:
            start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
            repeats=32 if stage=="walk" else 4
            start.record()
            for _ in range(repeats):graphs[a].replay()
            stop.record();stop.synchronize()
            times[a].append(start.elapsed_time(stop)*1000/repeats)
    return dict(times=times,orders=orders,wrong=wrong,changed=changed)
def main(stage,mode):
    seed=SEEDS[stage];g,weights,model=prepare(seed,mode)
    tag="selector_direct_"+mode
    scope="Single SM86 original captured _SelectorDraftSampler.__call__ including cuda RNG, Triton walk and all public buffer writes. "
    scope+=("Immutable fixture-supplied score lattice; five-layer draft, shared LM head and TP top-k excluded." if mode=="walk" else
        "Original rank-256 hidden projection, native _score_edges ATen arithmetic and real selector lattice build from changing global top-16 IDs; eager ATen lowered inside CUDA graph, no Inductor fusion, draft layers, LM head or TP collective.")
    canonical=dict(agent="speculation",candidate=tag,scope=scope,baseline_sha256=BASE,candidate_sha256=hashlib.sha256((D/"direct_candidate.diff").read_bytes()).hexdigest(),
        active=True,correct=True,complete=False,seed=seed,stage=stage,unit="CUDA event us per graph replay",bench=dict(instances=1,shapes={}))
    detail={}
    for b in (1,4,16,32):
        for risk in ("greedy","mixed"):
            name=f"B{b}_{risk}";v=run(b,seed,mode,risk,g,weights,model)
            detail[name]=v;canonical["bench"]["shapes"][name]=dict(weight=1,times=v["times"])
            canonical["correct"]&=all(n==0 for d in v["wrong"].values() for n in d.values()) and min(v["changed"])>0
            canonical["complete"]=len(detail)==8
            (D/f"{tag}_{stage}.json").write_text(json.dumps(canonical,indent=2))
            (D/f"{tag}_{stage}_detail.json").write_text(json.dumps(detail,indent=2))
            print(stage,mode,name,v["wrong"],"moved",min(v["changed"]),flush=True)
if __name__=="__main__":
    with torch.inference_mode():main(sys.argv[1],sys.argv[2])

