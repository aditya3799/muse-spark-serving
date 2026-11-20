"""Generate real greedy verifier continuations and record draft lattices at every eligible prefix."""
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
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from tokenizers import Tokenizer
from stream_pair import state, fetch, TARGET, HERE
from draft import Drafter

PROMPTS=[
    "Write a Python function first_duplicate(values) that returns the first repeated integer, or None.",
    "Why does binary search require the input list to be sorted? Give one counterexample.",
    "Implement a bounded queue in C++ using a vector, head, tail and count.",
    "Write SQL to list customers with more than one unpaid invoice in the last month.",
    "Explain what a failing unit test reveals when a parser drops a trailing empty field.",
    "Give a small TypeScript function that groups strings by their first character.",
    "Translate the sentence 'The engineer tested the new bridge' into Hindi and explain the verb.",
    "Write a git commit message for a fix that handles a closed socket during a partial write.",
    "Describe a two-pass algorithm to identify the majority element and verify the answer.",
    "Write Python code to undo a tree rotation while maintaining the parent pointers.",
    "Calculate the number of integer lattice points on the boundary of the rectangle with corners (0,0) and (5,3).",
    "Write a concise code review for a cache that deletes items while iterating over its dictionary.",
]
def generate(model,tokenizer,text,count):
    prompt="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n"
    values=tokenizer.encode(prompt,add_special_tokens=False).ids
    initial=len(values)
    x=torch.tensor([values],device="cuda")
    cache=None; taps=[]
    for step in range(count):
        out=model(input_ids=x,past_key_values=cache,use_cache=True,output_hidden_states=True,logits_to_keep=1)
        taps.append(torch.cat([out.hidden_states[k+1][0] for k in (2,10,18,26)],-1))
        token=out.logits[0,-1].argmax()
        cache=out.past_key_values;x=token.reshape(1,1)
        val=int(token);values.append(val)
        if val in (151643,151645):break
    del out,cache
    return torch.tensor(values,device="cuda"),torch.cat(taps),initial
def observe(model,drafter,tokenizer,text,n,count):
    tokens,taps,start=generate(model,tokenizer,text,count)
    anchors=list(range(start,len(tokens)-7))
    if not anchors:return dict(prompt=n,generated=len(tokens)-start,anchors=0)
    output=[[],[],[]];mismatch=None
    for s in range(0,len(anchors),4):
        pos=anchors[s:s+4]
        data=drafter.block_logits(taps,tokens,pos)
        if s==0:
            # First anchor must be independent of future target features from other batched anchors.
            ref=drafter.block_logits(taps[:pos[0]],tokens[:pos[0]+1],pos[:1])
            mismatch=dict(top16_id_differences=int((ref[0][0]!=data[0][0]).sum()),
                          maximum_score_error=float((ref[1][0].float()-data[1][0].float()).abs().max()))
        for bucket,t in zip(output,data):bucket.append(t.cpu())
    chosen,scores,coverage=[torch.cat(v) for v in output]
    np.savez_compressed(HERE/f"trace_{n:02d}.npz",tokens=tokens.cpu().numpy(),anchors=np.array(anchors),
                        ids=chosen.numpy().astype(np.int32),scores=scores.float().numpy().astype(np.float16),
                        log_unary_mass=coverage.float().numpy(),start=np.array(start))
    return dict(prompt=n,generated=len(tokens)-start,anchors=len(anchors),batch_shape_check=mismatch,
                first_tokens=tokenizer.decode(tokens[start:start+12].tolist()),prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
def main():
    p=argparse.ArgumentParser();p.add_argument("--prompts",type=int,default=12);p.add_argument("--tokens",type=int,default=72)
    args=p.parse_args()
    tokenizer=Tokenizer.from_str(fetch(TARGET+"tokenizer.json").decode())
    cfg=Qwen3Config(**json.loads((HERE/"target_config.json").read_text()))
    cfg._attn_implementation="sdpa"
    with torch.device("meta"): model=Qwen3ForCausalLM(cfg)
    weights,small=state("cuda")
    model.load_state_dict(weights,strict=True,assign=True)
    model.model.rotary_emb=Qwen3RotaryEmbedding(cfg,device="cuda")
    model.eval()
    drafter=Drafter(small,model.model.embed_tokens.weight,model.lm_head.weight)
    result=dict(target=TARGET,draft="rasyosef/Qwen3-1.7B-DFlash2@c8b13ac1d5013995cdb9f80f362efc9c03319ec3",device=torch.cuda.get_device_name(),
                scope="Real Qwen3 greedy generations and matched pretrained DFlash2 proposal logits. This is an external surrogate for tree search; no Muse or H200 serving rate is measured.",rows=[])
    for i,text in enumerate(PROMPTS[:args.prompts]):
        item=observe(model,drafter,tokenizer,text,i,args.tokens)
        print(i,item,flush=True);result["rows"].append(item)
        (HERE/"observed.json").write_text(json.dumps(result,indent=2))
    import replay
    replay.main()
    from bench_tree import run_live
    choice=json.loads((HERE/"policy_replay.json").read_text())["selected"]
    run_live(model,drafter,{int(k):v for k,v in choice.items()})

if __name__=="__main__":
    with torch.inference_mode():main()