"""Time wider rooted verification with a duplicate parent and independent public coding contexts."""
import os
os.environ["USE_TF"]="0"
os.environ["USE_FLAX"]="0"
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from stream_pair import HERE,TARGET,state,fetch
from draft import Drafter
import bench_tree as bench
from cuda_planner import CompiledPlanner
from validate_planner import prefixes
from selector_tree import build_tree

def public_contexts(tokenizer,model):
    corpus=HERE.parents[2]/"research_2026-11-18_frontier/patronus_muse.jsonl"
    row_ids=[0,3,6,9,12,15]
    selected=[]
    with corpus.open(encoding="utf-8") as inp:
        for ix,line in enumerate(inp):
            if ix not in row_ids:continue
            row=json.loads(line)
            task=next(s["message"] for s in row["steps"] if s["source"]=="user" and isinstance(s["message"],str))
            first=tokenizer.encode("<|im_start|>user\n",add_special_tokens=False).ids
            tail=tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n",add_special_tokens=False).ids
            text=tokenizer.encode(task,add_special_tokens=False).ids
            cap=(256,512,768)[row_ids.index(ix)%3]-len(first)-len(tail)
            if len(text)>cap:text=text[:cap//2]+text[-(cap-cap//2):]
            prompt=first+text+tail
            gold=[];cache=None;x=torch.tensor([prompt],device="cuda")
            for _ in range(32):
                out=model(input_ids=x,past_key_values=cache,use_cache=True,logits_to_keep=1)
                x=out.logits[:,-1].argmax(-1).view(1,1);cache=out.past_key_values
                value=int(x[0,0]);gold.append(value)
                if value in (151643,151645):break
            del out,cache
            if len(gold)>8:
                selected.append((f"public_{ix}",prompt,gold,hashlib.sha256(task.encode()).hexdigest()))
    return selected

def check_planners(compiled,small,policies):
    d=dict(np.load(HERE/"trace_00.npz"));checked=0
    for index in (0,9,30,49):
        ids=torch.from_numpy(d["ids"][index:index+1]).long().cuda()
        scores=torch.from_numpy(d["scores"][index:index+1]).bfloat16().cuda()
        mass=torch.from_numpy(d["log_unary_mass"][index:index+1]).cuda()
        anchor=torch.tensor([d["tokens"][d["anchors"][index]]],device="cuda")
        base=small.chain(ids,scores,anchor)[0]
        assert int(base[0])==int(anchor[0])
        for width,planner in compiled.items():
            policy=policies[width]
            _,t,m,f=policy.split("_")
            ids0,p,depth,mask=planner.single(ids,scores,anchor,policy,mass)
            covered=mass.exp()[:,:,None].expand(-1,-1,16) if int(m[1:]) else None
            expected=build_tree(ids,scores.float()/float(t[1:]),anchor,width,covered,after_fork=float(f[1:]))
            # Quantized logits can tie. A different equally likely prefix must still have a valid rooted mask.
            assert torch.equal(mask.sum(-1)-1,depth.long()) and len(prefixes(ids0.tolist(),p))==width
            matched=prefixes(ids0.tolist(),p)==prefixes(expected[0][0].tolist(),expected[1][0].tolist())
            if not matched:raise AssertionError(("real lattice prefix set differs",width,index))
            checked+=1
    return checked

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--public",action="store_true")
    parser.add_argument("--cached-draft",action="store_true")
    args=parser.parse_args()
    policies={int(k):v["policy"] for k,v in json.loads((HERE/"policy_replay.json").read_text())["selected"].items()}
    small=CompiledPlanner("chain")
    trees={n:CompiledPlanner(f"planner{n}") for n in (12,16)}
    verified=check_planners(trees,small,policies)
    def propose(candidates,scores,mass,anchor,policy):
        if policy=="chain":return small.chain(candidates,scores,anchor)
        n=int(policy.split("_")[0][1:])
        return trees[n].single(candidates,scores,anchor,policy,mass)
    bench.propose=propose
    tokenizer=Tokenizer.from_str(fetch(TARGET+"tokenizer.json").decode())
    cfg=Qwen3Config(**json.loads((HERE/"target_config.json").read_text()));cfg._attn_implementation="sdpa"
    with torch.device("meta"): model=Qwen3ForCausalLM(cfg)
    w,short=state("cuda");model.load_state_dict(w,strict=True,assign=True)
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device="cuda");model.eval()
    from cached_draft import CachedDrafter
    cls=CachedDrafter if args.cached_draft else Drafter
    draft=cls(short,model.model.embed_tokens.weight,model.lm_head.weight);draft.use_parent_edges=True
    draft.inspect_first_error=True
    inputs=[]
    for ix in range(1,12,2):
        d=dict(np.load(HERE/f"trace_{ix:02d}.npz"));a=int(d["start"]);tokens=d["tokens"].tolist()
        inputs.append((f"held_{ix}",tokens[:a],tokens[a:a+40],None))
    if args.public:inputs.extend(public_contexts(tokenizer,model))
    out=dict(scope="Matched pretrained Qwen3/DFlash2 at one local request. True target forward, mask, DFlash, compiled chain/tree selection, accepted KV and hidden-state commit. Duplicate chain timing per prompt. Public coding-task prompts are truncated using fixed head+tail budgets; their targets are newly generated by Qwen. No official traffic or TP8 execution.",
             validated_compiled_sets=verified,train_source="frozen even short requests",policies=policies,requests=[])
    out["draft_context"]="Append committed context K/V once per accepted path" if args.cached_draft else "Re-evaluate the historical draft context projections in every forward"
    paths=HERE/("wide_cached_live.json" if args.cached_draft else "wide_live.json")
    first=inputs[0]
    for arm in ("chain",policies[12],policies[16]):bench.measured(model,draft,first[1],first[2][:8],arm)
    for i,(label,prefix,gold,sha) in enumerate(inputs):
        arms=["chain",policies[12],"control",policies[16]]
        if args.cached_draft:arms.extend([policies[12]+"_guard",policies[16]+"_guard"])
        if i%2:arms.reverse()
        r=(i//2)%len(arms);arms=arms[r:]+arms[:r]
        row=dict(name=label,context=len(prefix),tokens=len(gold),source_message_sha256=sha,arms={})
        for arm in arms:
            row["arms"][arm]=bench.measured(model,draft,prefix,gold,"chain" if arm=="control" else arm)
        ref=row["arms"]["chain"]["produced"]
        row["pair_token_differences"]={k:sum(a!=b for a,b in zip(ref,d["produced"])) for k,d in row["arms"].items()}
        out["requests"].append(row);paths.write_text(json.dumps(out,indent=2))
        print(label,{k:(len(d["steps"]),d["wrong"],round(d["seconds"],2)) for k,d in row["arms"].items()},flush=True)
if __name__=="__main__":
    with torch.inference_mode():main()
